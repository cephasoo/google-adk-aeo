import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json
import re # We need this for robust parsing
from google.cloud import secretmanager
from googleapiclient.discovery import build

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")

# --- Global Clients (Lazy Initialization) ---
model = None
search_api_key = None

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

@functions_framework.http
def handle_request(request):
    """The main agent orchestrator."""
    global model
    if model is None:
        print("Initializing Vertex AI model...")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        print("Model initialized.")

    request_json = request.get_json(silent=True)
    if not request_json or 'topic' not in request_json:
        return jsonify({"error": "Invalid request. JSON payload with a 'topic' key is required."}), 400

    topic = request_json['topic']
    print(f"Orchestrator starting for topic: {topic}")

    try:
        # --- Agent Workflow ---
        keyword_context = find_trending_keywords(topic)
        linked_concepts_data = create_euphemistic_links(keyword_context)
        story = tell_then_and_now_story(linked_concepts_data['interlinked_concepts'])
        
        final_output = {
            "story": story,
            "analysis": linked_concepts_data
        }
        return jsonify(final_output)

    except Exception as e:
        # This will now catch our new ValueError and provide a more useful debug message.
        print(f"An unexpected error occurred in the orchestrator: {e}")
        return jsonify({"error": str(e)}), 500