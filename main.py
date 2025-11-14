import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os

# --- Configuration (Loaded from environment variables) ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash-001")

# --- Global Client (Lazy Initialization) ---
# Initialize the model as None. We will create it only on the first request.
model = None

@functions_framework.http
def handle_request(request):
    """ Handles incoming HTTP requests with lazy initialization of the model. """
    
    global model  # Declare that we are using the global 'model' variable

    # --- This is the key change ---
    # If the model hasn't been initialized yet, do it now.
    # This block will only run ONCE, during the very first request to the function.
    if model is None:
        print("Initializing Vertex AI model for the first time...")
        if not PROJECT_ID or not LOCATION:
            raise EnvironmentError("GCP_PROJECT_ID and GCP_LOCATION environment variables must be set.")
        
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        print("Model initialized successfully.")

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