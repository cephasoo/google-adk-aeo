import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json
import re
import uuid
import requests
from bs4 import BeautifulSoup
from google.cloud import secretmanager, firestore, tasks_v2
from googleapiclient.discovery import build
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
BROWSERLESS_API_KEY = os.environ.get("BROWSERLESS_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
QUEUE_NAME = "story-worker-queue"
MAX_LOOP_ITERATIONS = 2 

PROPOSAL_KEYWORDS = ["outline", "draft", "proposal", "story", "brief"]

STORY_WORKER_URL = f"https://{LOCATION}-{PROJECT_ID}.cloudfunctions.net/process-story-logic"
FEEDBACK_WORKER_URL = f"https://{LOCATION}-{PROJECT_ID}.cloudfunctions.net/process-feedback-logic"

FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL", f"aeo-devops-agent-identity@{PROJECT_ID}.iam.gserviceaccount.com")

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
            "oidc_token": {"service_account_email": FUNCTION_IDENTITY_EMAIL}
        }
    }
    tasks_client.create_task(request={"parent": parent, "task": task})

def classify_feedback_intent(feedback_text):
    global model
    prompt = f"""Classify User Feedback: "{feedback_text}". Respond ONLY with: APPROVE or REFINE."""
    response = model.generate_content(prompt).text.strip().upper()
    if "APPROVE" in response: return "APPROVE"
    return "REFINE"

def extract_url_from_text(text):
    """Finds and cleans the first URL in a string, handling Slack formatting."""
    # Regex to capture http/s URLs, ignoring surrounding < > or |
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match:
        return match.group(1)
    return None


def fetch_article_content(url):
    """
    Robustly scrapes content using Browserless REST API.
    Fixes: Moves 'stealth' to Query Param, uses 'gotoOptions' for waiting.
    """
    print(f"Tool: Reading URL via Browserless: {url}")
    
    if not BROWSERLESS_API_KEY:
        return "Error: Browserless API Key is missing."

    # FIX 1: Pass 'stealth' as a Query Parameter, not in the body
    endpoint = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_API_KEY}&stealth=true"
    
    # FIX 2: Correct JSON Schema for /content endpoint
    payload = {
        "url": url,
        "rejectResourceTypes": ["image", "media", "font"], # Optimization: Block junk
        "gotoOptions": {
            "timeout": 15000, # 15s timeout
            "waitUntil": "networkidle2" # Wait for network to settle (handles dynamic pages)
        }
    }
    
    headers = {
        "Cache-Control": "no-cache", 
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=20)
        
        if response.status_code != 200:
            print(f"Browserless Error {response.status_code}: {response.text}")
            return f"Failed to read content. (Status: {response.status_code})"

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Aggressive Cleaning
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "ad"]):
            tag.decompose()

        text = soup.get_text(separator='\n\n')
        clean_text = re.sub(r'\n\s*\n', '\n\n', text).strip()
        
        if len(clean_text) < 200:
            return "Content appears to be behind a login wall or empty."

        return clean_text[:20000]

    except requests.exceptions.Timeout:
        return "Error: The page took too long to load."
    except Exception as e:
        print(f"Scraper Exception: {e}")
        return "Error: Could not read page due to an unexpected error."
    
# --- AI Tools ---
# --- UPDATED TOOL: Find Trending Keywords (URL-First Logic) ---
def find_trending_keywords(unstructured_topic, history_context=""):
    global model
    print(f"Tool: find_trending_keywords for input: '{unstructured_topic[:100]}...'")
    
    # 1. Prioritize URL Extraction
    url = extract_url_from_text(unstructured_topic)
    
    context_snippets = []
    clean_topic = ""
    
    if url:
        print(f"URL Detected: {url}. Switching to Grounding Mode.")
        article_text = fetch_article_content(url)
        
        # Format specifically for Grounding
        context_snippets = [f"GROUNDING_SOURCE_URL: {url}", f"GROUNDING_CONTENT:\n{article_text}"]
        
        # Logic change: If we have a URL, the "clean_topic" should be the user's intent
        # concerning that URL, or we let the LLM derive it later.
        # For now, we trust the Reader Tool.
        return {"context": context_snippets, "clean_topic": unstructured_topic, "is_grounded": True}
            
    else:
        print("No URL found. Using standard search mode.")
        extraction_prompt = f"""
        User Input: "{unstructured_topic}"
        Previous Context: "{history_context}"
        Task: Convert this input into a specific Google Search Query.
        Respond with ONLY the search query string.
        """
        search_query = model.generate_content(extraction_prompt).text.strip()
        print(f"Generated Search Query: {search_query}")

        api_key = get_search_api_key()
        service = build("customsearch", "v1", developerKey=api_key)
        
        res = service.cse().list(q=f"{search_query}", cx=SEARCH_ENGINE_ID, num=10).execute()
        google_snippets = [r.get('snippet', '') for r in res.get('items', [])]
        context_snippets = google_snippets
        clean_topic = search_query

    return {"context": context_snippets, "clean_topic": clean_topic}

def generate_comprehensive_answer(topic, context):
    global model
    print("Tool: Answer Generator")

    # 1. Determine Mode
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    
    # 2. Set Configuration Variables based on Mode
    if is_grounded:
        print("Mode: STRICT GROUNDING (Temp 0.0)")
        temperature = 0.0
        system_instruction = """
        CRITICAL INSTRUCTION: You are in READING MODE. 
        You have been provided with scraped content from a specific URL.
        You MUST base your answer PRIMARILY on the provided 'GROUNDING_CONTENT'.
        Do not hallucinate facts not present in the text.
        If the text does not contain the answer, explicitly state that.
        """
    else:
        print("Mode: RESEARCH ASSISTANT (Temp 0.7)")
        temperature = 0.7
        system_instruction = "You are an expert AI assistant and a master of communication. Use the research context to provide accurate answers."

    # 3. Construct the Shared Prompt
    # We inject the specific 'system_instruction' chosen above
    prompt = f"""
    {system_instruction}
    
    The user asked: "{topic}"
    
    Provide a comprehensive, direct, and helpful answer.
    If you include a table, you MUST use clean, standard Markdown syntax.
    
    Research Context:
    {context}
    
    Answer:
    """

    # 4. Execute Single Call
    response = model.generate_content(
        prompt, 
        generation_config={"temperature": temperature}
    )
    return response.text.strip()

def create_euphemistic_links(keyword_context):
    global model
    print("Tool: Writer")
    prompt = f"""
    Topic: "{keyword_context['clean_topic']}". Context: {keyword_context['context']}
    Identify 4-10 core keyword clusters for 'Then' and 6-10 for 'Now'.
    Create a euphemistic link between them.
    CRITICAL SCHEMA: Exact keys: "then_concept", "now_concept", "link".
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
    REWRITE proposal. Topic: {topic}. Draft: {json.dumps(current_proposal)}. Critique: {critique}. Preserve keys: 'then_concept', 'now_concept', 'link'.
    """
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
    
    # --- FIX: DETERMINISTIC ROUTING ---
    # Check for URL *before* asking the LLM. 
    # If a URL is present, FORCE 'WORK' mode to trigger the Scraper.
    has_url = extract_url_from_text(text_input) is not None
    
    intent = "SOCIAL" # Default
    
    if has_url:
        print(f"Routing: URL detected -> Forcing WORK mode.")
        intent = "WORK"
    else:
        # Fallback to LLM classification for non-URL text
        triage_prompt = f"Classify: '{text_input}'. Classes: [SOCIAL, WORK]. Respond ONLY with class name."
        intent = model.generate_content(triage_prompt).text.strip().upper()
    
    if "SOCIAL" in intent:
        social_prompt = f"User: '{text_input}'. Respond naturally/wittily. Max 2 sentences."
        reply = model.generate_content(social_prompt).text.strip()
        db.collection('agent_sessions').document(session_id).set({
            "status": "completed", "type": "social", "topic": "Social", "slack_context": slack_context,
            "event_log": [{"event_type": "social", "text": text_input}, {"event_type": "agent_reply", "text": reply}]
        })
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']})
        return jsonify({"msg": "Social reply sent", "session_id": session_id}), 200
    else:
        db.collection('agent_sessions').document(session_id).set({
             "status": "starting", "type": "work_answer", "topic": text_input, "slack_context": slack_context, "event_log": []
        })
        dispatch_task({"session_id": session_id, "topic": text_input, "slack_context": slack_context}, STORY_WORKER_URL)
        return jsonify({"type": "work", "msg": "Accepted", "session_id": session_id}), 202

@functions_framework.http
def handle_feedback_workflow(request):
    req = request.get_json(silent=True)
    dispatch_task({"session_id": req['session_id'], "feedback": req['feedback']}, FEEDBACK_WORKER_URL)
    return jsonify({"msg": "Feedback accepted"}), 202

# --- FUNCTION: Ingest Knowledge ---
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

    tagging_prompt = f"""
    Analyze story about "{topic}". Generate 5-10 searchable keywords.
    Response JSON format: {{ "keywords": ["tag1", "tag2"] }}
    STORY: {final_story[:2000]}
    """
    try:
        metadata = extract_json(model.generate_content(tagging_prompt).text)
    except Exception:
        metadata = {"keywords": [topic or "story"]}

    db.collection('knowledge_base').document(session_id).set({
        "topic": topic, "content": final_story, "keywords": metadata.get('keywords', []),
        "created_at": datetime.datetime.now(datetime.timezone.utc), "source_session": session_id
    })
    return jsonify({"msg": "Knowledge ingested."}), 200

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
        # 1. Reactive Triage
        is_proposal_request = any(re.search(rf"\b{w}\b", topic.lower()) for w in PROPOSAL_KEYWORDS)
        print(f"Keyword Triage for '{topic}': Proposal={is_proposal_request}")

        # 2. Research
        research_data = find_trending_keywords(topic)
        clean_topic = research_data['clean_topic']
        
        # FIX: Define the new events as a list to append
        new_events = [{"event_type": "user_request", "text": topic, "timestamp": str(datetime.datetime.now())}]

        if not is_proposal_request:
            # --- PATH A: Direct Answer ---
            answer_text = generate_comprehensive_answer(topic, research_data['context'])
            
            # Add answer to the batch
            new_events.append({"event_type": "agent_answer", "text": answer_text})
            
            # CORRECT: Uses update + ArrayUnion
            db.collection('agent_sessions').document(session_id).update({
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "topic": clean_topic,
                "slack_context": slack_context, 
                "event_log": firestore.ArrayUnion(new_events)
            })
            
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": answer_text,
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts']
            })
            return jsonify({"msg": "Answer sent"}), 200

        else:
            # --- PATH B: Proposal ---
            current_proposal = create_euphemistic_links(research_data)
            new_events.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})

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
            new_events.append({"event_type": "adk_request_confirmation", "approval_id": approval_id, "payload": final_proposal['interlinked_concepts']})
            
            # --- THE FIX WAS APPLIED HERE ---
            # CHANGED: set(..., merge=True) -> update(...)
            # CHANGED: event_log: new_events -> event_log: firestore.ArrayUnion(new_events)
            db.collection('agent_sessions').document(session_id).update({
                "status": "awaiting_approval", 
                "type": "work_proposal", 
                "topic": clean_topic,
                "slack_context": slack_context, 
                "event_log": firestore.ArrayUnion(new_events)
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

# --- WORKER 2: Feedback Logic (Improved Topic Extraction) ---
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
    session_doc = doc_ref.get()
    if not session_doc.exists: return jsonify({"error": "Session not found"}), 404
    session_data = session_doc.to_dict()
    slack_context = session_data.get('slack_context', {})
    
    session_type = session_data.get('type', 'work') 

    # 1. Conversational Handling (Social OR Previous Answer)
    if session_type in ['social', 'work_answer']:
        
        # --- NEW: SKILL ROUTER LOGIC ---
        # Check 1: Is this a URL/Research Request?
        url_in_feedback = extract_url_from_text(user_feedback)
        
        # Check 2: Is this a "Graduation" (Proposal) Request?
        is_graduation = any(w in user_feedback.lower() for w in PROPOSAL_KEYWORDS)
        
        # DECISION: Delegate to Story Worker if either is true
        if url_in_feedback or is_graduation:
            print(f"Delegation Triggered. URL={bool(url_in_feedback)}, Graduation={is_graduation}")
            
            target_topic = ""
            
            if url_in_feedback:
                # Case A: URL Found -> The input IS the topic (let Story Worker scrape it)
                target_topic = user_feedback
                new_status = "working_on_grounding"
                # We update type to 'work_answer' to keep it in research mode
                doc_ref.update({"type": "work_answer", "status": new_status})
                
            else:
                # Case B: Graduation -> Use LLM to extract the abstract topic
                history = session_data.get('event_log', [])
                history_text = "\n".join([f"{e.get('event_type')}: {e.get('text') or e.get('message') or e.get('result')}" for e in history[-5:]])
                
                ext_prompt = f"""
                The user wants to convert the current conversation into a formal proposal.
                HISTORY: {history_text}
                USER COMMAND: "{user_feedback}"
                Identify the SUBJECT MATTER. Ignore the command "draft this".
                Respond with ONLY the topic name.
                """
                target_topic = model.generate_content(ext_prompt).text.strip()
                doc_ref.update({"type": "work_proposal", "topic": target_topic, "status": "awaiting_approval"})

            # EXECUTE HANDOFF: Dispatch to the Specialist (Worker 1)
            # We pass the same session_id so it appends to the correct history
            dispatch_task({
                "session_id": session_id, 
                "topic": target_topic, 
                "slack_context": slack_context
            }, STORY_WORKER_URL)
            
            return jsonify({"msg": "Delegated to Research Worker"}), 200

        # --- END SKILL ROUTER ---

        # Normal Conversation (No URL, No Proposal command)
        history = session_data.get('event_log', [])
        context_text = "\n".join([f"{e.get('event_type')}: {e.get('text') or e.get('message')}" for e in history[-7:]])
        
        reply = model.generate_content(f"Reply to user in context:\n{context_text}\nUser: {user_feedback}").text.strip()
        
        doc_ref.update({"event_log": firestore.ArrayUnion([{"event_type": "user_feedback", "text": user_feedback}, {"event_type": "agent_reply", "text": reply}])})
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "type": "social", "message": reply,
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')
        })
        return jsonify({"msg": "reply sent"}), 200

    # 2. Work Proposal Handling (The Loop) - No changes needed here
    intent = classify_feedback_intent(user_feedback)
    user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": intent, "timestamp": str(datetime.datetime.now())}

    if intent == "APPROVE":
        pending_event = next((e for e in reversed(session_data.get('event_log', [])) if e.get('event_type') == 'adk_request_confirmation'), None)
        if not pending_event: return jsonify({"error": "No proposal found."}), 500
        
        story = tell_then_and_now_story(pending_event['payload'], tool_confirmation={"confirmed": True})
        
        doc_ref.update({"status": "completed", "final_story": story, "event_log": firestore.ArrayUnion([user_event, {"event_type": "story", "content": story}])})
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "proposal": [{"link": story}],
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True
        })
        
    elif intent == "QUESTION":
        answer_text = generate_comprehensive_answer(user_feedback, "User pivoting from proposal to question.")
        doc_ref.update({
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

    return jsonify({"message": "Feedback processed."}), 200