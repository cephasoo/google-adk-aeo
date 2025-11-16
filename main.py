import functions_framework
from flask import jsonify
from google.cloud import run_v2 # The new client library for Cloud Run
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")

# This agent is now single-purpose: it lists Cloud Run services.
# In the future, we would add the Gemini model back in to *decide* which tool to run.

@functions_framework.http
def handle_request(request):
    
    if not PROJECT_ID or not LOCATION:
        raise EnvironmentError("GCP_PROJECT_ID and GCP_LOCATION environment variables must be set.")

    try:
        # --- Agent Step 1: Initialize the Tool ---
        # Create a client to interact with the Cloud Run API.
        print("Initializing Cloud Run client.")
        client = run_v2.ServicesClient()

        # --- Agent Step 2: Execute the Tool ---
        # Prepare the request to list services. The 'parent' is the location to search in.
        parent_location = f"projects/{PROJECT_ID}/locations/{LOCATION}"
        print(f"Listing services in: {parent_location}")
        
        request = run_v2.ListServicesRequest(
            parent=parent_location,
        )

        # Make the API call
        page_result = client.list_services(request=request)

        # --- Agent Step 3: Format the Output ---
        # Process the response and create a simple list of service names.
        services_list = []
        for service in page_result:
            services_list.append(service.name)
        
        print(f"Found {len(services_list)} services.")
        return jsonify({"services": services_list})

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500