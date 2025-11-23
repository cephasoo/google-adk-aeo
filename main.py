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
    """Robustly extracts JSON object from text using Regex."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: raise ValueError(f"No JSON found: {text}")
    return json.loads(match.group(0))

def dispatch_task(payload, target_url):
    """Helper to send a payload to a Cloud Task Worker."""
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
    print(f"Classifying intent: {feedback_text}")
    prompt = f"""
    You are managing a content creation workflow.
    Classify User Feedback: "{feedback_text}"
    Classes:
    1. APPROVE: Likes it, wants to finalize (e.g., "Looks good", "Yes").
    2. REFINE: Wants changes (e.g., "Make it shorter").
    3. QUESTION: Stopping to ask a general question, NOT editing (e.g., "Why did this happen?").
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
    extraction_prompt = f"""
    The user sent this message: "{unstructured_topic}".
    Understand their INTENT and convert it into a Google Search Query.
    Respond with ONLY the search query string.
    """
    search_query = model.generate_content(extraction_prompt).text.strip()
    
    api_key = get_search_api_key()
    service = build("customsearch", "v1", developerKey=api_key)
    res_then = service.cse().list(q=f"history traditional concepts {search_query}", cx=SEARCH_ENGINE_ID, num=10).execute()
    res_now = service.cse().list(q=f"modern trends state {search_query}", cx=SEARCH_ENGINE_ID, num=10).execute()
    
    snippets_then = [r.get('snippet', '') for r in res_then.get('items', [])]
    snippets_now = [r.get('snippet', '') for r in res_now.get('items', [])]
    
    return {"context": snippets_then + snippets_now, "clean_topic": search_query}

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
    print("Tool: Refiner")
    prompt = f"""
    REWRITE proposal. Topic: {topic}. Draft: {json.dumps(current_proposal)}. Critique: {critique}. 
    CRITICAL SCHEMA: Preserve exact keys: "then_concept", "now_concept", "link".
    """
    return extract_json(model.generate_content(prompt).text)

def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None):
    global model
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    prompt = f"Tell a 'Then and Now' story using: {interlinked_concepts}"
    return model.generate_content(prompt).text


# --- DISPATCHER 1: START ---
@functions_framework.http
def start_story_workflow(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
    
    if db is None: db = firestore.Client()

    request_json = request.get_json(silent=True)
    text_input = request_json.get('topic') 
    
    # FIX: Bundle Slack Context into a dictionary here
    slack_context = { 
        "ts": request_json.get('slack_ts'), 
        "thread_ts": request_json.get('slack_thread_ts'), 
        "channel": request_json.get('slack_channel') 
    }

    if not text_input: return jsonify({"error": "Invalid request."}), 400

    session_id = str(uuid.uuid4())

    # Triage
    triage_prompt = f"""
    Classify message: "{text_input}"
    Classes: [SOCIAL, WORK]
    SOCIAL: Greetings, small talk, statements like "Life is good".
    WORK: Story requests, topics, questions about specific subjects.
    Respond ONLY with class name.
    """
    intent = model.generate_content(triage_prompt).text.strip().upper()
    print(f"Session {session_id} Triage: {intent}")

    if "SOCIAL" in intent:
        social_prompt = f"""
        User Input: "{text_input}"
        Role: Witty, concise AI storyteller.
        Task: Respond naturally. Then, briefly invite them to provide a topic for a "Then vs Now" story.
        Constraints: Max 2 sentences. No lists.
        """
        reply = model.generate_content(social_prompt).text.strip()
        
        session_ref = db.collection('agent_sessions').document(session_id)
        session_ref.set({
            "status": "completed",
            "type": "social",
            "topic": "Social Interaction",
            "slack_context": slack_context,
            "event_log": [
                {"event_type": "user_request", "text": text_input, "timestamp": datetime.datetime.now(datetime.timezone.utc)},
                {"event_type": "agent_reply", "text": reply, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
            ]
        })
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, 
            "type": "social",
            "message": reply,
            "channel_id": slack_context['channel'],
            "thread_ts": slack_context['ts']
        })
        
        return jsonify({"message": "Social reply sent.", "session_id": session_id}), 200

    else:
        # WORK
        print(f"Dispatching WORK task: {session_id}")
        db.collection('agent_sessions').document(session_id).set({
             "status": "starting", "type": "work", "topic": text_input, "slack_context": slack_context, "event_log": []
        })
        dispatch_task({
            "session_id": session_id, "topic": text_input, "slack_context": slack_context
        }, STORY_WORKER_URL)
        
        return jsonify({"type": "work", "message": "Workflow accepted.", "session_id": session_id}), 202

# --- DISPATCHER 2: FEEDBACK ---
@functions_framework.http
def handle_feedback_workflow(request):
    req = request.get_json(silent=True)
    dispatch_task({"session_id": req['session_id'], "feedback": req['feedback']}, FEEDBACK_WORKER_URL)
    return jsonify({"msg": "Feedback accepted"}), 202

# --- Dispatcher 3: Ingest Knowledge (Long-Term Memory) ---
@functions_framework.http
def ingest_knowledge(request):
    global db, model
    if db is None: db = firestore.Client()
    if model is None:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)

    request_json = request.get_json(silent=True)
    session_id = request_json.get('session_id')
    final_story = request_json.get('story')
    topic = request_json.get('topic')
    
    if not final_story: return jsonify({"error": "Missing story"}), 400

    # Generate Keywords
    tagging_prompt = f"""
    Analyze story about "{topic}". Generate 5-10 searchable keywords.
    Response JSON format: {{ "keywords": ["tag1", "tag2"] }}
    STORY: {final_story[:2000]}
    """
    try:
        metadata = extract_json(model.generate_content(tagging_prompt).text)
    except Exception:
        metadata = {"keywords": [topic or "story"]}

    # Save to Knowledge Base
    db.collection('knowledge_base').document(session_id).set({
        "topic": topic,
        "content": final_story,
        "keywords": metadata.get('keywords', []),
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "source_session": session_id
    })
    return jsonify({"msg": "Knowledge ingested."}), 200

# --- WORKER 1: STORY LOGIC ---
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
        # 1. Research
        research_data = find_trending_keywords(topic)
        clean_topic = research_data['clean_topic']
        
        # 2. Reactive Triage (Proposal vs Answer)
        is_proposal_request = any(w in topic.lower() for w in PROPOSAL_KEYWORDS)
        print(f"Keyword Triage for '{topic}': Proposal={is_proposal_request}")

        event_log = [
            {"event_type": "user_request", "text": topic, "timestamp": str(datetime.datetime.now())},
            {"event_type": "research", "clean_topic": clean_topic}
        ]
        
        if not is_proposal_request:
            # PATH A: Direct Answer
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

# --- WORKER 2: FEEDBACK LOGIC ---
@functions_framework.http
def process_feedback_logic(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()

    request_json = request.get_json(silent=True)
    session_id = request_json['session_id']
    user_feedback = request_json['feedback']
    
    # Standardized variable name: session_ref
    session_ref = db.collection('agent_sessions').document(session_id)
    session_doc = session_ref.get()
    if not session_doc.exists: return jsonify({"error": "Session not found"}), 404
    session_data = session_doc.to_dict()
    slack_context = session_data.get('slack_context', {})

    session_type = session_data.get('type', 'work') 

    if session_type in ['social', 'work_answer']:
        # 1. Semantic Check: Is this a pivot to work?
        transition_prompt = f"""
        User Input: "{user_feedback}"
        Determine if this is a request for a specific TOPIC/STORY (WORK) or just CHAT (SOCIAL).
        Respond with JSON: {{ "intent": "SOCIAL" or "WORK", "topic": "The extracted topic if WORK, else null" }}
        """
        try:
            transition_response = extract_json(model.generate_content(transition_prompt).text)
            intent = transition_response.get("intent", "SOCIAL")
            new_topic = transition_response.get("topic")
        except Exception:
            intent = "SOCIAL"

        # 2. Handle Graduation (Social -> Work)
        if intent == "WORK" and new_topic:
            print(f"Graduating Session {session_id} to WORK. Topic: {new_topic}")
            session_ref.update({
                "type": "work_proposal", "topic": new_topic, "status": "awaiting_approval" 
            })
            # Dispatch to Story Worker with "Story Proposal:" prefix to force Path B
            dispatch_task({
                "session_id": session_id, "topic": f"Story Proposal: {new_topic}", "slack_context": slack_context
            }, STORY_WORKER_URL)
            return jsonify({"message": "Session graduated to Work."}), 200

        # 3. Handle Social Continuation
        else:
            history = session_data.get('event_log', [])
            conversation_context = "\n".join([f"{'AI' if e.get('event_type') == 'agent_reply' else 'User'}: {e.get('text') or e.get('message')}" for e in history[-7:]])

            chat_prompt = f"""
            Friendly AI storyteller. Conversation: {conversation_context}. User: "{user_feedback}".
            Task: Respond directly and briefly. No lists. No meta-commentary.
            """
            reply = model.generate_content(chat_prompt).text.strip()
            
            session_ref.update({
                "event_log": firestore.ArrayUnion([
                    {"event_type": "user_feedback", "text": user_feedback, "timestamp": datetime.datetime.now(datetime.timezone.utc)},
                    {"event_type": "agent_reply", "text": reply, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
                ])
            })
            
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, "type": "social", "message": reply,
                "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')
            })
            return jsonify({"message": "Social reply processed."}), 200

    # --- WORK Logic ---
    intent = classify_feedback_intent(user_feedback)
    user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": intent, "timestamp": str(datetime.datetime.now())}

    if intent == "APPROVE":
        pending_event = next((e for e in reversed(session_data.get('event_log', [])) if e.get('event_type') == 'adk_request_confirmation'), None)
        if not pending_event: return jsonify({"error": "No proposal found."}), 500
        
        story = tell_then_and_now_story(pending_event['payload'], tool_confirmation={"confirmed": True})
        
        session_ref.update({"status": "completed", "final_story": story, "event_log": firestore.ArrayUnion([user_event, {"event_type": "story", "content": story}])})
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "proposal": [{"link": story}],
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True
        })

    elif intent == "QUESTION":
        answer_text = generate_comprehensive_answer(user_feedback, "User pivoting from proposal to question.")
        session_ref.update({
            "type": "work_answer", "status": "awaiting_feedback",
            "event_log": firestore.ArrayUnion([user_event, {"event_type": "agent_answer", "text": answer_text}])
        })
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "type": "social", "message": answer_text,
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')
        })

    else: # REFINE
        last_prop = next((e for e in reversed(session_data.get('event_log', [])) if e.get('proposal_data')), None)
        if not last_prop: return jsonify({"error": "No proposal"}), 500
        try:
            new_prop = refine_proposal(session_data.get('topic'), last_prop['proposal_data'], user_feedback)
        except Exception: return jsonify({"error": "JSON failed"}), 500
        new_id = f"approval_{uuid.uuid4().hex[:8]}"
        session_ref.update({
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

    return jsonify({"message": "Feedback processed."}), 200