import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
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
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") # To send proposals to Slack
APPROVAL_KEYWORDS = ["looks good", "proceed", "approved", "continue", "yes"]

# --- Global Clients ---
model = None
search_api_key = None
db = None

def get_search_api_key():
    """Securely fetches the Google Search API key from Secret Manager."""
    global search_api_key
    if search_api_key is None:
        print("Fetching Google Search API Key from Secret Manager...")
        client = secretmanager.SecretManagerServiceClient()
        secret_name = "google-search-api-key"
        resource_name = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(name=resource_name)
        search_api_key = response.payload.data.decode("UTF-8")
        print("API Key fetched successfully.")
    return search_api_key

# --- Tool 1: Find Trending Keywords ---
def find_trending_keywords(topic):
    """Uses Google Search to find relevant keywords for a topic."""
    print(f"Tool: find_trending_keywords, Topic: {topic}")
    api_key = get_search_api_key()
    service = build("customsearch", "v1", developerKey=api_key)
    query_then = f"traditional SEO concepts for {topic}"
    query_now = f"modern AEO agentic optimization concepts for {topic}"
    
    res_then = service.cse().list(q=query_then, cx=SEARCH_ENGINE_ID, num=5).execute()
    res_now = service.cse().list(q=query_now, cx=SEARCH_ENGINE_ID, num=5).execute()
    
    snippets_then = [result.get('snippet', '') for result in res_then.get('items', [])]
    snippets_now = [result.get('snippet', '') for result in res_now.get('items', [])]
    
    return {"then_keywords_context": snippets_then, "now_keywords_context": snippets_now}

# --- Tool 2: Create Euphemistic Links (UPDATED AND ROBUST) ---
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
    
    # --- This is the key change for robustness ---
    # We find the JSON block within the model's text, even if it adds extra words.
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a valid JSON object in the model's response. Response text: {response.text}")
        
    json_text = match.group(0)
    return json.loads(json_text) # Only parse the extracted JSON

# --- Tool 3: Tell a "Then and Now" Story ---
def tell_then_and_now_story(interlinked_concepts):
    """Uses the LLM to write a story based on the linked concepts."""
    print("Tool: tell_then_and_now_story")
    prompt = f"""
    Tell a short, compelling "Then and Now" themed story using the following interlinked concepts.
    The story should flow naturally, using the 'Then' concepts as the past and the 'Now' concepts as the present or future, linked by the euphemisms.

    Interlinked Concepts: {interlinked_concepts}
    """
    response = model.generate_content(prompt)
    return response.text

# --- NEW FUNCTION 1: Starts the Workflow ---
@functions_framework.http
def start_story_workflow(request):
    """Receives user topic, starts the agent, and presents the first proposal."""
    global model, db
    if model is None: # Initialize clients
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        db = firestore.Client()

    request_json = request.get_json(silent=True)
    topic = request_json.get('topic') if request_json else None
    if not topic:
        return jsonify({"error": "Invalid request. JSON with 'topic' key required."}), 400

    session_id = str(uuid.uuid4())
    print(f"Starting workflow for session: {session_id}, topic: {topic}")

    try:
        # Agent's First Actions
        keyword_context = find_trending_keywords(topic)
        first_proposal = create_euphemistic_links(keyword_context)

        # Create session memory in Firestore
        session_ref = db.collection('agent_sessions').document(session_id)
        session_data = {
            "status": "awaiting_feedback",
            "topic": topic,
            "history": [
                {"role": "agent", "proposal": first_proposal, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
            ]
        }
        session_ref.set(session_data)
        
        # Pause and Request Human Feedback via N8N Webhook
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id,
            "proposal": first_proposal['interlinked_concepts']
        })
        
        return jsonify({"message": "Workflow started. Awaiting feedback.", "session_id": session_id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW FUNCTION 2: Handles the Conversational Loop ---
@functions_framework.http
def handle_feedback_workflow(request):
    """Receives feedback from N8N, refines the proposal, or finalizes the story."""
    global model, db
    if model is None: # Ensure clients are warm
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
        # Optionally, send the final story back to Slack via N8N
        return jsonify({"message": "Workflow completed successfully."}), 200

    # --- If not approved, it's a refinement request ---
    else:
        print(f"Refinement request for session {session_id}: {user_feedback}")
        last_proposal = session_data['history'][-1]['proposal']

        # Construct a new prompt for refinement
        refinement_prompt = f"""
        A user has provided feedback on a set of proposed concepts. Your task is to revise the proposal based on this feedback.
        
        PREVIOUS PROPOSAL:
        {json.dumps(last_proposal, indent=2)}

        USER FEEDBACK:
        "{user_feedback}"

        Please generate a new, revised proposal that incorporates the user's feedback.
        Your response MUST be a single, valid JSON object in the exact same format as the original proposal. Do not add any conversational text.
        """
        
        revised_proposal_text = model.generate_content(refinement_prompt).text
        match = re.search(r'\{.*\}', revised_proposal_text, re.DOTALL)
        if not match:
            # Handle cases where the model fails to generate valid JSON
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
            "proposal": revised_proposal['interlinked_concepts']
        })

        return jsonify({"message": "Proposal revised and sent for another review."}), 200