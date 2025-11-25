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
    prompt = f"""Classify User Feedback: "{feedback_text}". Respond ONLY with: APPROVE or REFINE."""
    response = model.generate_content(prompt).text.strip().upper()
    if "APPROVE" in response: return "APPROVE"
    return "REFINE"

def extract_url_from_text(text):
    """Finds the first URL in a string."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None


def fetch_article_content(url):
    """
    Uses Browserless.io to scrape content from protected sites.
    """
    global model
    print(f"Tool: Reading URL via Browserless: {url}")
    
    # 1. Get the secret API key
    client = secretmanager.SecretManagerServiceClient()
    key_name = f"projects/{PROJECT_ID}/secrets/browserless-api-key/versions/latest"
    key_response = client.access_secret_version(name=key_name)
    api_key = key_response.payload.data.decode("UTF-8")
    
    # 2. Construct the Browserless API call
    browserless_url = f"https://chrome.browserless.io/content?token={api_key}"
    
    payload = {
        "url": url,
        "waitFor": 1000 # Wait 1 second for page to load JS
    }
    
    try:
        response = requests.post(browserless_url, json=payload, timeout=30)
        response.raise_for_status()
        
        # 3. Use BeautifulSoup to clean the raw HTML returned by Browserless
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        if len(text) < 100:
            return "Could not extract meaningful content from the page."
        
        # 4. Use Gemini to Summarize the scraped text (It will be noisy)
        summary_prompt = f"""
        This is raw text scraped from a webpage. Summarize the main points into a clean, readable article.
        SCRAPED TEXT:
        {text[:15000]}
        """
        summary = model.generate_content(summary_prompt).text.strip()
        return summary

    except Exception as e:
        print(f"Error reading URL via Browserless: {e}")
        return f"Failed to read the article at {url}. The site may be heavily protected."

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
        # PATH A: READ MODE
        print(f"URL Detected: {url}. Switching to Reader Mode.")
        
        # Call the new Browserless function
        article_text = fetch_article_content(url)
        
        context_snippets.append(f"[SOURCE ARTICLE]: {article_text}")
        
        # Use LLM to extract topic FROM the summary
        topic_extraction_prompt = f"""
        Analyze this article summary. Identify the main topic.
        SUMMARY: {article_text[:1000]}...
        Respond with ONLY the topic name.
        """
        clean_topic = model.generate_content(topic_extraction_prompt).text.strip()
        print(f"Extracted Topic from URL Content: {clean_topic}")
            
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
    """
    Writes a natural answer and, if useful, provides a structured summary as a JSON object.
    This separates content generation from presentation.
    """
    global model
    print("Tool: Answer Generator")
    prompt = f"""
    You are an expert AI assistant and a master of communication. The user asked: "{topic}"
    
    Use the following research context to provide a comprehensive, direct, and helpful answer in natural paragraphs.
    Choose the BEST format for any summary, based on the content (e.g., bullet points, paragraphs, or a table).

    If you choose to create a table, you MUST use clean, standard Markdown syntax.
    
    ---
    EXAMPLE of a good Markdown table:
    | Header 1 | Header 2 |
    | :--- | :--- |
    | Row 1, Cell 1 | Row 1, Cell 2 |
    | Row 2, Cell 1 | Row 2, Cell 2 |
    ---
    
    Context: {context}

    Answer:
    """
    response = model.generate_content(prompt)
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
        event_log = [{"event_type": "user_request", "text": topic, "timestamp": str(datetime.datetime.now())}]

        if not is_proposal_request:
            # --- PATH A: Direct Answer ---
            # The tool now returns a single Markdown string
            answer_text = generate_comprehensive_answer(topic, research_data['context'])
            
            # We log the raw text, letting N8N handle the parsing
            event_log.append({"event_type": "agent_answer", "text": answer_text})
            
            db.collection('agent_sessions').document(session_id).set({
                "status": "awaiting_feedback", "type": "work_answer", "topic": clean_topic,
                "slack_context": slack_context, "event_log": event_log
            }, merge=True)
            
            # We now send a payload that matches what the "Social" type expects in N8N
            # N8N will parse this for a table and render it correctly.
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", # Tell N8N to treat this as a simple text/markdown message
                "message": answer_text,
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts']
            })
            return jsonify({"msg": "Answer sent"}), 200

        else:
            # --- PATH B: Proposal ---
            # This logic remains the same
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
            }, merge=True)

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
        
        is_graduation = any(w in user_feedback.lower() for w in PROPOSAL_KEYWORDS)
        
        if is_graduation:
            history = session_data.get('event_log', [])
            history_text = "\n".join([f"{e.get('event_type')}: {e.get('text') or e.get('message') or e.get('result')}" for e in history[-5:]])
            
            # --- THE FIX: SMARTER PROMPT ---
            ext_prompt = f"""
            The user wants to convert the current conversation into a formal "Then vs Now" story proposal.
            
            CONVERSATION HISTORY:
            {history_text}
            
            USER'S LAST COMMAND:
            "{user_feedback}"
            
            TASK: Identify the SUBJECT MATTER we have been discussing (e.g., "Voice Search", "Coffee", "SEO").
            CRITICAL: Ignore the user's command to "make a draft" or "write a story." Look at the CONTENT of the previous turns.
            
            Respond with ONLY the topic name.
            """
            derived_topic = model.generate_content(ext_prompt).text.strip()
            print(f"Extracted Graduation Topic: {derived_topic}")

            doc_ref.update({"type": "work_proposal", "topic": derived_topic, "status": "awaiting_approval"})
            
            dispatch_task({
                "session_id": session_id, "topic": f"Story Proposal: {derived_topic}", "slack_context": slack_context
            }, STORY_WORKER_URL)
            return jsonify({"msg": "Graduated"}), 200

        # Normal Conversation
        history = session_data.get('event_log', [])
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