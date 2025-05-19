# File: test_supabase_connection.py

import os
from supabase import create_client, Client # Make sure 'Client' is imported for type hinting
import logging

# --- For local development: Load environment variables from .env file ---
# Make sure you have a .env file in the same directory (d:\ai_toxdata_engine)
# with your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
# Example .env content:
# SUPABASE_URL="https://your-project-ref.supabase.co"
# SUPABASE_SERVICE_ROLE_KEY="your-very-long-service-role-key"
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("INFO: Loaded environment variables from .env file (if it exists).")
except ImportError:
    print("INFO: python-dotenv library not found. Will rely on system environment variables.")
# --- End of .env loading ---


# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def init_supabase_client() -> Client | None:
    """
    Initializes and returns a Supabase client instance using environment variables.
    This client will use the service_role key for administrative operations.
    """
    supabase_url: str | None = os.environ.get("SUPABASE_URL")
    supabase_service_key: str | None = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url:
        logging.error("CRITICAL: SUPABASE_URL environment variable not set.")
        return None
    if not supabase_service_key:
        logging.error("CRITICAL: SUPABASE_SERVICE_ROLE_KEY environment variable not set.")
        return None

    try:
        # Create the Supabase client
        supabase: Client = create_client(supabase_url, supabase_service_key)
        logging.info("Successfully initialized Supabase client object.")
        return supabase
    except Exception as e:
        logging.error(f"Failed to initialize Supabase client object: {e}")
        return None

if __name__ == "__main__":
    logging.info("--- Starting Supabase Connection Test ---")
    
    # Attempt to initialize the Supabase client
    client = init_supabase_client()

    if client:
        logging.info("Supabase client object initialized successfully.")
        logging.info("Attempting a simple test query to verify connection...")
        try:
            # Perform a simple query.
            # We'll try to select from 'toxicology_data' table.
            # If the table doesn't exist, it's still a sign the connection worked but the query failed.
            # If the connection itself failed, client would be None or this would raise a different error.
            response = client.table("toxicology_data").select("test_cas", count="exact").limit(1).execute()

            # Check response structure (supabase-py v2.x has .data, .error, .count attributes on APIResponse)
            if hasattr(response, 'error') and response.error:
                # Check for specific error: table not found
                if "relation" in str(response.error.message).lower() and "does not exist" in str(response.error.message).lower():
                    logging.info(f"CONNECTION TEST SUCCESSFUL: Connected to Supabase, but the table 'toxicology_data' does not exist yet. This is okay for a connection test. Error: {response.error.message}")
                else:
                    logging.error(f"CONNECTION TEST FAILED (API Error): Supabase API error during test query: {response.error.message} (Code: {response.error.code}, Details: {response.error.details})")
            elif hasattr(response, 'data'):
                logging.info(f"CONNECTION TEST SUCCESSFUL: Test query executed successfully. Number of records found (limit 1): {len(response.data)}, Total potential count: {response.count}")
            else:
                # Fallback for older supabase-py or unexpected response structure
                logging.warning(f"CONNECTION TEST UNCERTAIN: Test query executed, but response structure was not as expected. Response: {response}")

        except Exception as e:
            logging.error(f"CONNECTION TEST FAILED (Exception): An error occurred during the Supabase test query: {e}", exc_info=True)
    else:
        logging.critical("CONNECTION TEST FAILED: Supabase client object could not be initialized. Check environment variables (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) and previous logs.")
    
    logging.info("--- Supabase Connection Test Finished ---")