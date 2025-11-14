import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os # Best Practice: Import os to handle environment variables

# --- Configuration (Loaded from environment variables) ---
# This makes your code portable and secure.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash-001") # Provide a sensible default

# --- Initialize Vertex AI ---
# This check ensures the function won't start without its configuration.
if not PROJECT_ID or not LOCATION:
    raise EnvironmentError("GCP_PROJECT_ID and GCP_LOCATION environment variables must be set.")

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)

@functions_framework.http
def handle_request(request):
    """ Handles incoming HTTP requests. """
    request_json = request.get_json(silent=True)

    if not request_json or 'prompt' not in request_json:
        return jsonify({"error": "Invalid request. JSON payload with a 'prompt' key is required."}), 400

    prompt = request_json['prompt']
    print(f"Received prompt: {prompt}")

    try:
        # --- Call the Gemini API ---
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048
            )
        )
        
        agent_response = response.candidates[0].content.parts[0].text
        print(f"Agent response: {agent_response}")
        
        return jsonify({"response": agent_response})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": str(e)}), 500