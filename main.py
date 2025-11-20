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

# We define TWO worker URLs now
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
        secret_name = "google-search-api-key"
        resource_name = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(name=resource_name)
        search_api_key = response.payload.data.decode("UTF-8")
    return search_api_key

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a valid JSON object in response. Text: {text}")
    return json.loads(match.group(0))

def classify_feedback_intent(feedback_text):
    global model
    print(f"Classifying feedback intent: {feedback_text}")
    prompt = f"""
    You are an AI workflow manager. Determine if the user is APPROVING the proposal or requesting CHANGES.
    User Message: "{feedback_text}"
    Respond with ONLY one word: APPROVE or REFINE.
    """
    response = model.generate_content(prompt).text.strip().upper()
    if "APPROVE" in response: return "APPROVE"
    return "REFINE"

# --- Tools ---
def find_trending_keywords(unstructured_topic):
    global model
    extraction_prompt = f"""
    The user provided the following query: "{unstructured_topic}".
    Your task is to identify and extract the CORE, concise topic area for a "Then vs Now" story.
    Respond with ONLY the core topic area, no conversational text.
    """
    extracted_topic = model.generate_content(extraction_prompt).text.strip()
    
    api_key = get_search_api_key()
    service = build("customsearch", "v1", developerKey=api_key)
    query_then = f"traditional concepts for {extracted_topic}"
    query_now = f"modern agentic optimization for {extracted_topic}"
    res_then = service.cse().list(q=query_then, cx=SEARCH_ENGINE_ID, num=10).execute()
    res_now = service.cse().list(q=query_now, cx=SEARCH_ENGINE_ID, num=10).execute()
    snippets_then = [result.get('snippet', '') for result in res_then.get('items', [])]
    snippets_now = [result.get('snippet', '') for result in res_now.get('items', [])]
    return {"then_keywords_context": snippets_then, "now_keywords_context": snippets_now, "clean_topic": extracted_topic}

def create_euphemistic_links(keyword_context):
    global model
    prompt = f"""
    Based on the following context, identify 4-10 core keyword clusters for the 'Then' (traditional) concepts and 6-10 for the 'Now' (modern) concepts.
    Then, create a euphemistic or metaphorical link between each 'Then' cluster and a corresponding 'Now' cluster.
    Your response MUST be a single, valid JSON object.
    CRITICAL SCHEMA REQUIREMENT: The "interlinked_concepts" array MUST use these EXACT keys: "then_concept", "now_concept", "link".
    Structure: {{ "then_clusters": [], "now_clusters": [], "interlinked_concepts": [ {{ "then_concept": "...", "now_concept": "...", "link": "..." }} ] }}
    'Then' Context: {keyword_context['then_keywords_context']}
    'Now' Context: {keyword_context['now_keywords_context']}
    """
    response = model.generate_content(prompt)
    return extract_json(response.text)

def critique_proposal(topic, current_proposal):
    global model
    prompt = f"""
    You are a critical editor. Review the following "Then vs Now" concept map for the topic: "{topic}".
    PROPOSAL: {json.dumps(current_proposal, indent=2)}
    INSTRUCTIONS: If the proposal is excellent, respond with exactly: APPROVED. If there are weak points, provide specific feedback.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def refine_proposal(topic, current_proposal, critique):
    global model
    prompt = f"""
    You are an expert editor. REWRITE the proposal to address the critique.
    TOPIC: {topic}
    DRAFT: {json.dumps(current_proposal, indent=2)}
    CRITIQUE: {critique}
    Your response MUST be a single, valid JSON object in the exact same structure.
    CRITICAL SCHEMA REQUIREMENT: Preserve the exact keys: "then_concept", "now_concept", "link".
    """
    response = model.generate_content(prompt)
    return extract_json(response.text)

def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None):
    global model
    if not tool_confirmation or not tool_confirmation.get("confirmed"):
        raise PermissionError("Cannot generate story without explicit human approval.")
    prompt = f"""
    A human has approved the following concepts. Tell a short, compelling "Then and Now" themed story.
    Interlinked Concepts: {interlinked_concepts}
    """
    response = model.generate_content(prompt)
    return response.text

# --- HELPER: Cloud Task Dispatcher ---
def dispatch_task(payload, target_url):
    global tasks_client
    if tasks_client is None:
        tasks_client = tasks_v2.CloudTasksClient()
    
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

# --- FUNCTION 1: Start Dispatcher ---
@functions_framework.http
def start_story_workflow(request):
    """Receives N8N request, dispatches task, returns 202 immediately."""
    request_json = request.get_json(silent=True)
    topic = request_json.get('topic')
    if not topic: return jsonify({"error": "Invalid request."}), 400

    session_id = str(uuid.uuid4())
    # Dispatch to STORY WORKER
    dispatch_task({
        "session_id": session_id, "topic": topic,
        "slack_context": {
            "ts": request_json.get('slack_ts'),
            "thread_ts": request_json.get('slack_thread_ts'),
            "channel": request_json.get('slack_channel')
        }
    }, STORY_WORKER_URL)
    
    return jsonify({"message": "Workflow accepted.", "session_id": session_id}), 202

# --- FUNCTION 2: Feedback Dispatcher (NEW) ---
@functions_framework.http
def handle_feedback_workflow(request):
    """Receives Feedback, dispatches task, returns 202 immediately."""
    request_json = request.get_json(silent=True)
    session_id = request_json.get('session_id')
    if not session_id: return jsonify({"error": "Invalid request."}), 400

    # Dispatch to FEEDBACK WORKER
    dispatch_task({
        "session_id": session_id,
        "feedback": request_json.get('feedback')
    }, FEEDBACK_WORKER_URL)

    return jsonify({"message": "Feedback accepted.", "session_id": session_id}), 202

# --- FUNCTION 3: Story Worker (Heavy Lifting) ---
@functions_framework.http
def process_story_logic(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()

    request_json = request.get_json(silent=True)
    session_id = request_json['session_id']
    topic = request_json['topic']
    slack_context = request_json['slack_context']

    try:
        event_log = []
        event_log.append({"event_type": "user_request", "text": topic, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
        
        keyword_context = find_trending_keywords(topic)
        clean_topic = keyword_context['clean_topic']
        event_log.append({"event_type": "tool_call", "tool_name": "find_trending_keywords", "result_summary": f"Found context for '{clean_topic}'"})
        
        # Writer-Critic Loop
        current_proposal = create_euphemistic_links(keyword_context)
        # FIX: Standardize key to 'proposal_data'
        event_log.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})

        loop_count = 0
        final_proposal = current_proposal

        MAX_ITERATIONS = 2
        while loop_count < MAX_ITERATIONS:
            critique = critique_proposal(clean_topic, current_proposal)
            event_log.append({"event_type": "loop_critique", "iteration": loop_count, "critique": critique})
            
            if "APPROVED" in critique: break
            
            try:
                current_proposal = refine_proposal(clean_topic, current_proposal, critique)
                final_proposal = current_proposal
                # FIX: Standardize key to 'proposal_data'
                event_log.append({"event_type": "loop_refinement", "iteration": loop_count, "proposal_data": final_proposal})
            except ValueError: break 
            loop_count += 1

        approval_id = f"approval_{uuid.uuid4().hex[:8]}"
        request_confirmation_event = {
            "event_type": "adk_request_confirmation",
            "approval_id": approval_id,
            "hint": "Please review.",
            "payload": final_proposal['interlinked_concepts']
        }
        event_log.append(request_confirmation_event)
        
        session_ref = db.collection('agent_sessions').document(session_id)
        session_ref.set({
            "status": "awaiting_approval",
            "topic": clean_topic,
            "slack_context": slack_context,
            "event_log": event_log
        })

        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, 
            "approval_id": approval_id,
            "proposal": final_proposal['interlinked_concepts'],
            "thread_ts": slack_context['ts'], 
            "channel_id": slack_context['channel'],
            "is_initial_post": True 
        })
        return jsonify({"message": "Worker finished."}), 200
    except Exception as e:
        print(f"Story Worker failed: {e}")
        return jsonify({"error": str(e)}), 500

# --- FUNCTION 4: Feedback Worker (Heavy Lifting) ---
@functions_framework.http
def process_feedback_logic(request):
    """Handles the AI reasoning for feedback (Approve vs Refine)."""
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()

    request_json = request.get_json(silent=True)
    session_id = request_json['session_id']
    user_feedback = request_json['feedback']

    print(f"Feedback Worker for session: {session_id}")
    
    session_ref = db.collection('agent_sessions').document(session_id)
    session_doc = session_ref.get()
    if not session_doc.exists: return jsonify({"error": "Session not found"}), 404
    session_data = session_doc.to_dict()
    slack_context = session_data.get('slack_context', {})

    # 1. Classify Intent (AI Task)
    intent = classify_feedback_intent(user_feedback)
    print(f"Intent classified as: {intent}")
    
    is_approved = (intent == "APPROVE")
    user_feedback_event = {
        "event_type": "user_feedback", "text": user_feedback, "intent": intent, "timestamp": datetime.datetime.now(datetime.timezone.utc)
    }

    if is_approved:
        # Find the last request for confirmation
        pending_event = next((event for event in reversed(session_data.get('event_log', [])) if event.get('event_type') == 'adk_request_confirmation'), None)
        if not pending_event: return jsonify({"error": "No proposal to approve."}), 500
        
        concepts = pending_event['payload']
        
        # 2. Write Story (AI Task)
        story = tell_then_and_now_story(concepts, tool_confirmation={"confirmed": True})
        
        tool_conf_event = {"event_type": "adk_tool_confirmation", "confirmed": True}
        story_event = {"event_type": "tool_call_result", "tool_name": "tell_then_and_now_story", "story": story}
        
        session_ref.update({
            "status": "completed", "final_story": story,
            "event_log": firestore.ArrayUnion([user_feedback_event, tool_conf_event, story_event])
        })
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id,
            "proposal": [{"link": story}],
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'),
            "is_final_story": True
        })
        
    else: # Refine
        # FIX: Find last event with 'proposal_data' (consistent key now)
        last_proposal_event = next((event for event in reversed(session_data.get('event_log', [])) if event.get('proposal_data')), None)
        if not last_proposal_event: return jsonify({"error": "No previous proposal data found."}), 500
        
        last_proposal = last_proposal_event['proposal_data']
        
        # 2. Refine Proposal (AI Task)
        refinement_prompt = f"""
        A user has provided feedback. Revise the proposal.
        PREVIOUS PROPOSAL: {json.dumps(last_proposal, indent=2)}
        USER FEEDBACK: "{user_feedback}"
        Generate a new, revised proposal in the exact same JSON format.
        CRITICAL SCHEMA REQUIREMENT: You MUST preserve the exact keys: "then_concept", "now_concept", and "link".
        """
        revised_proposal_text = model.generate_content(refinement_prompt).text
        try:
            revised_proposal = extract_json(revised_proposal_text)
        except Exception: return jsonify({"error": "JSON generation failed"}), 500

        new_proposal_event = {"event_type": "agent_proposal", "proposal_data": revised_proposal}
        new_approval_id = f"approval_{uuid.uuid4().hex[:8]}"
        new_req_event = {"event_type": "adk_request_confirmation", "approval_id": new_approval_id, "payload": revised_proposal['interlinked_concepts']}
        
        session_ref.update({
            "event_log": firestore.ArrayUnion([user_feedback_event, new_proposal_event, new_req_event])
        })
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, "approval_id": new_approval_id,
            "proposal": revised_proposal['interlinked_concepts'],
            "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')
        })

    return jsonify({"message": "Feedback processed."}), 200