import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json
import re
import uuid
import requests
from google.cloud import secretmanager, firestore, tasks_v2
from googleapiclient.discovery import build
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
QUEUE_NAME = "story-worker-queue"
MAX_LOOP_ITERATIONS = 2 
PROPOSAL_KEYWORDS = ["outline", "draft", "proposal", "story", "brief"]

STORY_WORKER_URL = f"https://{LOCATION}-{PROJECT_ID}.cloudfunctions.net/process-story-logic"
FEEDBACK_WORKER_URL = f"https://{LOCATION}-{PROJECT_ID}.cloudfunctions.net/process-feedback-logic"

# --- Global Clients ---
model = None
search_api_key = None
db = None
tasks_client = None

# --- Utility Functions ---
def get_search_api_key():
    global search_api_key
    if search_api_key is None:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(name=f"projects/{PROJECT_ID}/secrets/google-search-api-key/versions/latest")
        search_api_key = response.payload.data.decode("UTF-8")
    return search_api_key

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: raise ValueError(f"No JSON found: {text}")
    return json.loads(match.group(0))

def dispatch_task(payload, target_url):
    global tasks_client
    if tasks_client is None: tasks_client = tasks_v2.CloudTasksClient()
    parent = tasks_client.queue_path(PROJECT_ID, LOCATION, QUEUE_NAME)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": target_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload).encode(),
            "oidc_token": {"service_account_email": f"aeo-devops-agent-identity@{PROJECT_ID}.iam.gserviceaccount.com"}
        }
    }
    tasks_client.create_task(request={"parent": parent, "task": task})

def classify_feedback_intent(feedback_text):
    global model
    prompt = f"""
    You are managing a content creation workflow. The user has sent a message regarding a proposal.
    Classify their intent into one of three categories:
    1. APPROVE: They like it and want to finalize/proceed.
    2. REFINE: They want to change specific parts of the proposal.
    3. QUESTION: They are stopping to ask a general question or discuss the topic, NOT editing the draft.
    User Message: "{feedback_text}"
    Respond ONLY with: APPROVE, REFINE, or QUESTION.
    """
    response = model.generate_content(prompt).text.strip().upper()
    if "APPROVE" in response: return "APPROVE"
    if "QUESTION" in response: return "QUESTION"
    return "REFINE"

# --- AI Tools ---

def find_trending_keywords(unstructured_topic):
    global model
    print(f"Tool: find_trending_keywords for '{unstructured_topic}'")
    extraction_prompt = f"Extract core topic/entity from: '{unstructured_topic}'. Respond ONLY with the topic."
    extracted_topic = model.generate_content(extraction_prompt).text.strip()
    
    api_key = get_search_api_key()
    service = build("customsearch", "v1", developerKey=api_key)
    q_query = f"{extracted_topic} history vs current state"
    
    res = service.cse().list(q=q_query, cx=SEARCH_ENGINE_ID, num=10).execute()
    snippets = [r.get('snippet', '') for r in res.get('items', [])]
    
    return {"context": snippets, "clean_topic": extracted_topic}

def generate_comprehensive_answer(topic, context):
    global model
    print("Tool: Answer Generator")
    prompt = f"""
    You are an expert AI assistant. The user asked: "{topic}"
    Use the following research context to provide a comprehensive, direct, and helpful answer.
    Do NOT use a "Then vs Now" list format. Write in natural paragraphs.
    Context: {context}
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def create_euphemistic_links(keyword_context):
    global model
    print("Tool: Writer (Proposal)")
    user_intent = keyword_context['clean_topic']
    prompt = f"""
    Topic: "{user_intent}". Context: {keyword_context['context']}
    Identify 4-10 core keyword clusters for 'Then' and 6-10 for 'Now'.
    Create a euphemistic link between them.
    CRITICAL SCHEMA: The "interlinked_concepts" array MUST use EXACT keys: "then_concept", "now_concept", "link".
    Structure: {{ "then_clusters": [], "now_clusters": [], "interlinked_concepts": [ {{ "then_concept": "...", "now_concept": "...", "link": "..." }} ] }}
    """
    response = model.generate_content(prompt)
    return extract_json(response.text)

def critique_proposal(topic, current_proposal):
    global model
    prompt = f"Review proposal for '{topic}': {json.dumps(current_proposal, indent=2)}. If excellent, respond: APPROVED. Else, feedback."
    return model.generate_content(prompt).text.strip()

def refine_proposal(topic, current_proposal, critique):
    global model
    prompt = f"REWRITE proposal. Topic: {topic}. Draft: {json.dumps(current_proposal)}. Critique: {critique}. Preserve keys: 'then_concept', 'now_concept', 'link'."
    return extract_json(model.generate_content(prompt).text)

def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None):
    global model
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    prompt = f"Tell a 'Then and Now' story using: {interlinked_concepts}"
    return model.generate_content(prompt).text


# --- DISPATCHERS ---
@functions_framework.http
def start_story_workflow(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
    if db is None: db = firestore.Client()

    request_json = request.get_json(silent=True)
    text_input = request_json.get('topic')
    slack_context = { "ts": request_json.get('slack_ts'), "thread_ts": request_json.get('slack_thread_ts'), "channel": request_json.get('slack_channel') }

    if not text_input: return jsonify({"error": "Invalid request"}), 400
    
    session_id = str(uuid.uuid4())
    triage_prompt = f"Classify: '{text_input}'. Classes: [SOCIAL, WORK]. Respond ONLY with class name."
    intent = model.generate_content(triage_prompt).text.strip().upper()
    
    if "SOCIAL" in intent:
        reply = model.generate_content(f"Reply wittily: '{text_input}'").text.strip()
        db.collection('agent_sessions').document(session_id).set({
            "status": "completed", "type": "social", "topic": "Social", "slack_context": slack_context,
            "event_log": [{"event_type": "social", "text": text_input}]
        })
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']})
        return jsonify({"msg": "Social reply sent", "session_id": session_id}), 200
    else:
        db.collection('agent_sessions').document(session_id).set({
             "status": "starting", "type": "work", "topic": text_input, "slack_context": slack_context, "event_log": []
        })
        dispatch_task({"session_id": session_id, "topic": text_input, "slack_context": slack_context}, STORY_WORKER_URL)
        return jsonify({"type": "work", "msg": "Accepted", "session_id": session_id}), 202

@functions_framework.http
def handle_feedback_workflow(request):
    req = request.get_json(silent=True)
    dispatch_task({"session_id": req['session_id'], "feedback": req['feedback']}, FEEDBACK_WORKER_URL)
    return jsonify({"msg": "Feedback accepted"}), 202


# --- WORKER 1: Story Logic ---
@functions_framework.http
def process_story_logic(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()

    req = request.get_json(silent=True)
    session_id = req['session_id']
    topic = req['topic']
    slack_context = req['slack_context']

    try:
        research_data = find_trending_keywords(topic)
        clean_topic = research_data['clean_topic']
        
        # Reactive Triage
        is_proposal_request = any(w in topic.lower() for w in PROPOSAL_KEYWORDS)
        print(f"Keyword Triage: Proposal={is_proposal_request}")

        event_log = [{"event_type": "user_request", "text": topic, "timestamp": str(datetime.datetime.now())}]
        
        if not is_proposal_request:
            # PATH A: Answer
            answer_text = generate_comprehensive_answer(topic, research_data['context'])
            event_log.append({"event_type": "agent_answer", "text": answer_text})
            
            db.collection('agent_sessions').document(session_id).set({
                "status": "awaiting_feedback", "type": "work_answer", "topic": clean_topic,
                "slack_context": slack_context, "event_log": event_log
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, "type": "social", "message": answer_text,
                "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']
            })
            return jsonify({"msg": "Answer sent"}), 200

        else:
            # PATH B: Proposal
            current_proposal = create_euphemistic_links(research_data)
            event_log.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})

            loop_count = 0
            final_proposal = current_proposal
            while loop_count < MAX_LOOP_ITERATIONS:
                critique = critique_proposal(clean_topic, current_proposal)
                if "APPROVED" in critique: break
                try:
                    final_proposal = refine_proposal(clean_topic, current_proposal, critique)
                    current_proposal = final_proposal
                except Exception: break
                loop_count += 1
                
            approval_id = f"approval_{uuid.uuid4().hex[:8]}"
            event_log.append({"event_type": "adk_request_confirmation", "approval_id": approval_id, "payload": final_proposal['interlinked_concepts']})
            
            db.collection('agent_sessions').document(session_id).set({
                "status": "awaiting_approval", "type": "work_proposal", "topic": clean_topic,
                "slack_context": slack_context, "event_log": event_log
            })

            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, "approval_id": approval_id,
                "proposal": final_proposal['interlinked_concepts'],
                "thread_ts": slack_context['ts'], "channel_id": slack_context['channel'],
                "is_initial_post": True 
            })
            return jsonify({"msg": "Proposal sent"}), 200

    except Exception as e:
        print(f"Worker Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- WORKER 2: Feedback Logic (FINAL CORRECTED) ---
@functions_framework.http
def process_feedback_logic(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()

    req = request.get_json(silent=True)
    session_id = req['session_id']
    user_feedback = req['feedback']
    
    doc_ref = db.collection('agent_sessions').document(session_id)
    data = doc_ref.get().to_dict()
    session_type = data.get('type', 'work')
    slack_context = data.get('slack_context', {})
    
    # 1. Conversational Handling (Social OR Previous Answer)
    if session_type in ['social', 'work_answer']:
        
        # GRADUATION CHECK
        is_graduation = any(w in user_feedback.lower() for w in PROPOSAL_KEYWORDS)
        
        if is_graduation:
            history = data.get('event_log', [])
            history_text = "\n".join([f"{e.get('event_type')}: {e.get('text') or e.get('message') or e.get('result')}" for e in history[-5:]])
            
            ext_prompt = f"""Extract TOPIC from history. History: {history_text}. Last Msg: "{user_feedback}". Respond ONLY with topic."""
            derived_topic = model.generate_content(ext_prompt).text.strip()

            doc_ref.update({"type": "work_proposal", "topic": derived_topic, "status": "awaiting_approval"})
            
            # Force Proposal Mode via Prefix
            dispatch_task({
                "session_id": session_id, "topic": f"Story Proposal: {derived_topic}", "slack_context": slack_context
            }, STORY_WORKER_URL)
            return jsonify({"msg": "Graduated"}), 200

        # Normal Conversation
        history = data.get('event_log', [])
        context_text = "\n".join([f"{e.get('event_type')}: {e.get('text') or e.get('message')}" for e in history[-7:]])
        reply = model.generate_content(f"Reply to user in context:\n{context_text}\nUser: {user_feedback}").text.strip()
        
        doc_ref.update({"event_log": firestore.ArrayUnion([{"event_type": "user_feedback", "text": user_feedback}, {"event_type": "agent_reply", "text": reply}])})
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "type": "social", "message": reply,
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')
        })
        return jsonify({"msg": "reply sent"}), 200

    # 2. Work Proposal Handling (The Loop)
    intent = classify_feedback_intent(user_feedback)
    user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": intent, "timestamp": str(datetime.datetime.now())}

    # PATH A: APPROVE
    if intent == "APPROVE":
        pending_event = next((e for e in reversed(data.get('event_log', [])) if e.get('event_type') == 'adk_request_confirmation'), None)
        if not pending_event: return jsonify({"error": "No proposal"}), 500
        
        story = tell_then_and_now_story(pending_event['payload'], tool_confirmation={"confirmed": True})
        
        doc_ref.update({"status": "completed", "final_story": story, "event_log": firestore.ArrayUnion([user_event, {"event_type": "story", "content": story}])})
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "proposal": [{"link": story}],
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True
        })

    # PATH B: QUESTION (Downgrade)
    elif intent == "QUESTION":
        # Downgrade to Answer Mode
        answer_text = generate_comprehensive_answer(user_feedback, "User pivoting from proposal to question.")
        
        doc_ref.update({
            "type": "work_answer", "status": "awaiting_feedback",
            "event_log": firestore.ArrayUnion([user_event, {"event_type": "agent_answer", "text": answer_text}])
        })
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "type": "social", "message": answer_text,
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')
        })

    # PATH C: REFINE
    else: 
        last_prop = next((e for e in reversed(data.get('event_log', [])) if e.get('proposal_data')), None)
        if not last_prop: return jsonify({"error": "No proposal"}), 500
        
        try:
            new_prop = refine_proposal(data['topic'], last_prop['proposal_data'], user_feedback)
        except Exception: return jsonify({"error": "JSON failed"}), 500

        new_id = f"approval_{uuid.uuid4().hex[:8]}"
        
        doc_ref.update({
            "event_log": firestore.ArrayUnion([
                user_event, 
                {"event_type": "agent_proposal", "proposal_data": new_prop},
                {"event_type": "adk_request_confirmation", "approval_id": new_id, "payload": new_prop['interlinked_concepts']}
            ])
        })
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "approval_id": new_id,
            "proposal": new_prop['interlinked_concepts'],
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')
        })

    return jsonify({"msg": "done"}), 200

# --- Worker 3: Knowledge Ingestion ---
@functions_framework.http
def ingest_knowledge(request):
    """
    Receives a finalized story and saves it to the permanent Knowledge Base.
    Triggered by N8N after the Google Doc is created.
    """
    global db, model
    if db is None: db = firestore.Client()
    if model is None:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)

    request_json = request.get_json(silent=True)
    session_id = request_json.get('session_id')
    final_story = request_json.get('story')
    topic = request_json.get('topic')
    
    if not final_story or not topic:
        return jsonify({"error": "Missing story or topic"}), 400

    print(f"Ingesting knowledge for topic: {topic}")

    # 1. Generate Searchable Tags/Summary (AI Task)
    # We ask the AI to distill the story into keywords for easier retrieval later.
    tagging_prompt = f"""
    Analyze this story about "{topic}".
    Generate a list of 5-10 searchable keywords or concepts discussed in the text.
    Also generate a 1-sentence summary.
    Response JSON format: {{ "keywords": [], "summary": "..." }}
    
    STORY:
    {final_story[:2000]}... # Truncate if too long to save tokens
    """
    try:
        metadata = extract_json(model.generate_content(tagging_prompt).text)
    except Exception:
        metadata = {"keywords": [topic], "summary": "Story generated by agent."}

    # 2. Save to Firestore 'knowledge_base' collection
    kb_ref = db.collection('knowledge_base').document(session_id)
    kb_ref.set({
        "topic": topic,
        "content": final_story,
        "keywords": metadata['keywords'],
        "summary": metadata['summary'],
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "source_session": session_id
    })

    return jsonify({"message": "Knowledge ingested successfully.", "tags": metadata['keywords']}), 200