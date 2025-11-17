import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, Content
import os
import json
import re
import uuid
import requests
from google.cloud import secretmanager, firestore
from googleapiclient.discovery import build
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
APPROVAL_KEYWORDS = ["looks good", "proceed", "approved", "continue", "yes"]

# --- Global Clients ---
model = None
search_api_key = None
db = None

# --- Utility Functions ---
def get_search_api_key():
    """Securely fetches the Google Search API key from Secret Manager."""
    global search_api_key
    if search_api_key is None:
        client = secretmanager.SecretManagerServiceClient()
        secret_name = "google-search-api-key"
        resource_name = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(name=resource_name)
        search_api_key = response.payload.data.decode("UTF-8")
    return search_api_key

# --- Tool 1: Find Trending Keywords (ENHANCED for Unstructured Topics) ---
def find_trending_keywords(unstructured_topic):
    """
    Uses the LLM to extract the core topic from an unstructured query, 
    then uses Google Search to find relevant 'then' and 'now' contexts.
    """
    global model
    print(f"Tool: find_trending_keywords, Unstructured Topic: {unstructured_topic}")

    # 1. Use LLM to extract a clean topic for searching
    extraction_prompt = f"""
    The user provided the following query: "{unstructured_topic}".
    Your task is to identify and extract the CORE, concise topic area for a "Then vs Now" story.
    Respond with ONLY the core topic area, no conversational text.
    Example Input: "What has changed about accessibility and structured headings?"
    Example Output: "Accessibility and Structured Headings"
    """
    
    extracted_topic = model.generate_content(extraction_prompt).text.strip()
    print(f"Extracted Core Topic: {extracted_topic}")

    # 2. Use the extracted topic for Google Search (Tool Execution)
    api_key = get_search_api_key()
    service = build("customsearch", "v1", developerKey=api_key)
    
    query_then = f"traditional concepts for {extracted_topic}"
    query_now = f"modern agentic optimization for {extracted_topic}"
    
    res_then = service.cse().list(q=query_then, cx=SEARCH_ENGINE_ID, num=5).execute()
    res_now = service.cse().list(q=query_now, cx=SEARCH_ENGINE_ID, num=5).execute()
    
    snippets_then = [result.get('snippet', '') for result in res_then.get('items', [])]
    snippets_now = [result.get('snippet', '') for result in res_now.get('items', [])]
    
    return {"then_keywords_context": snippets_then, "now_keywords_context": snippets_now, "clean_topic": extracted_topic}

# --- Tool 2: Create Euphemistic Links ---
def create_euphemistic_links(keyword_context):
    """Uses the LLM to cluster keywords and create euphemistic links."""
    print("Tool: create_euphemistic_links")
    prompt = f"""
    Based on the following context, identify 3-5 core keyword clusters for the 'Then' (traditional) concepts and 3-5 for the 'Now' (modern) concepts.
    Then, create a euphemistic or metaphorical link between each 'Then' cluster and a corresponding 'Now' cluster.
    Your response MUST be a single, valid JSON object and nothing else.
    The JSON object must contain two keys: "then_clusters" and "now_clusters", and a third key "interlinked_concepts".
    Do not include any conversational text or markdown formatting.

    'Then' Context: {keyword_context['then_keywords_context']}
    'Now' Context: {keyword_context['now_keywords_context']}
    """
    response = model.generate_content(prompt)
    
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a valid JSON object in the model's response. Response text: {response.text}")
        
    json_text = match.group(0)
    return json.loads(json_text)

# --- Tool 3: Tell a "Then and Now" Story ---
def tell_then_and_now_story(interlinked_concepts):
    """Uses the LLM to write a story based on the linked concepts."""
    print("Tool: tell_then_and-now_story")
    prompt = f"""
    Tell a short, compelling "Then and Now" themed story using the following interlinked concepts.
    The story should flow naturally, using the 'Then' concepts as the past and the 'Now' concepts as the present or future, linked by the euphemisms.

    Interlinked Concepts: {interlinked_concepts}
    """
    response = model.generate_content(prompt)
    return response.text

# --- NEW FUNCTION 1: Starts the Workflow (UPDATED for all Slack Context) ---
@functions_framework.http
def start_story_workflow(request):
    """Receives user topic and Slack context, starts the agent, and presents the first proposal."""
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()

    request_json = request.get_json(silent=True)
    topic = request_json.get('topic') if request_json else None
    
    # CRITICAL: Receive all Slack Context fields from N8N
    slack_ts = request_json.get('slack_ts')
    slack_thread_ts = request_json.get('slack_thread_ts') 
    slack_channel = request_json.get('slack_channel')
    
    # Check for required context fields. thread_ts can be None for a new message.
    if not topic or not slack_ts or not slack_channel:
        return jsonify({"error": "Invalid request. JSON with 'topic', 'slack_ts', and 'slack_channel' required."}), 400

    session_id = str(uuid.uuid4())
    print(f"Starting workflow for session: {session_id}, topic: {topic}")

    try:
        # Agent's First Actions: Handles UNSTRUCTURED topic
        keyword_context = find_trending_keywords(topic)
        first_proposal = create_euphemistic_links(keyword_context)

        # Create session memory in Firestore
        session_ref = db.collection('agent_sessions').document(session_id)
        session_data = {
            "status": "awaiting_feedback",
            "topic": keyword_context['clean_topic'], # Store the CLEANED topic
            
            # CRITICAL: Persist all Slack Context in Firestore
            "slack_ts": slack_ts,
            "slack_thread_ts": slack_thread_ts,
            "slack_channel": slack_channel,
            
            "history": [
                {"role": "agent", "proposal": first_proposal, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
            ]
        }
        session_ref.set(session_data)
        
        # Pause and Request Human Feedback via N8N Webhook
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id,
            "proposal": first_proposal['interlinked_concepts'],
            
            # Pass thread context back to N8N (Workflow 1) for clean posting
            # For a new message, slack_ts acts as the thread_ts for the reply
            "thread_ts": slack_ts, 
            "channel_id": slack_channel,
            "is_initial_post": True # New flag to help Workflow 1 distinguish replies
        })
        
        return jsonify({"message": "Workflow started. Awaiting feedback.", "session_id": session_id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW FUNCTION 2: Handles the Conversational Loop ---
@functions_framework.http
def handle_feedback_workflow(request):
    """Receives feedback from N8N, refines the proposal, or finalizes the story."""
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()
    
    request_json = request.get_json(silent=True)
    session_id = request_json.get('session_id') if request_json else None
    user_feedback = request_json.get('feedback') if request_json else None
    
    if not session_id or not user_feedback:
        return jsonify({"error": "Invalid request from N8N."}), 400

    print(f"Handling feedback for session: {session_id}")
    session_ref = db.collection('agent_sessions').document(session_id)
    session_doc = session_ref.get()

    if not session_doc.exists:
        return jsonify({"error": "Invalid session ID."}), 404

    session_data = session_doc.to_dict()
    
    # Retrieve Slack Context from Firestore for the N8N reply
    slack_channel = session_data.get('slack_channel')
    # Use the original slack_ts as the thread_ts for the reply
    slack_thread_ts = session_data.get('slack_ts') 

    # --- Check for Approval Keyword ---
    if any(keyword in user_feedback.lower() for keyword in APPROVAL_KEYWORDS):
        print(f"Approval detected for session {session_id}.")
        last_proposal = session_data['history'][-1]['proposal']
        story = tell_then_and_now_story(last_proposal['interlinked_concepts'])
        
        session_ref.update({
            "status": "completed",
            "final_story": story,
            "history": firestore.ArrayUnion([
                {"role": "user", "feedback": user_feedback, "timestamp": datetime.datetime.now(datetime.timezone.utc)},
                {"role": "agent", "final_story": story, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
            ])
        })
        # Final story posting request to N8N
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id,
            "proposal": [{"link": story}], # Send story as a simple list for N8N to handle
            "thread_ts": slack_thread_ts, 
            "channel_id": slack_channel,
            "is_final_story": True
        })
        return jsonify({"message": "Workflow completed successfully."}), 200

    # --- If not approved, it's a refinement request ---
    else:
        print(f"Refinement request for session {session_id}: {user_feedback}")
        last_proposal = session_data['history'][-1]['proposal']

        # Construct a new prompt for refinement (unchanged)
        refinement_prompt = f"""...""" # Prompt for Gemini...
        
        revised_proposal_text = model.generate_content(refinement_prompt).text
        match = re.search(r'\{.*\}', revised_proposal_text, re.DOTALL)
        if not match:
            return jsonify({"error": "Failed to generate a revised proposal."}), 500

        revised_proposal = json.loads(match.group(0))

        # Update history and send back to user for another review
        session_ref.update({
            "history": firestore.ArrayUnion([
                {"role": "user", "feedback": user_feedback, "timestamp": datetime.datetime.now(datetime.timezone.utc)},
                {"role": "agent", "proposal": revised_proposal, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
            ])
        })
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id,
            "proposal": revised_proposal['interlinked_concepts'],
            "thread_ts": slack_thread_ts, 
            "channel_id": slack_channel
        })

        return jsonify({"message": "Proposal revised and sent for another review."}), 200