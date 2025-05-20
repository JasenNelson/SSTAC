# File: test_db_ops.py

# Imports from your db_operations.py and init_supabase_client
# These assume 'db_operations.py' and 'test_supabase_connection.py' (or wherever init_supabase_client is)
# are in the SAME directory (d:\ai_toxdata_engine) or accessible in Python's path.

# It's cleaner if init_supabase_client is in a common utility file,
# but for now, if it's in test_supabase_connection.py, we import from there.
try:
    from test_supabase_connection import init_supabase_client
except ImportError:
    print("ERROR: Could not import 'init_supabase_client' from 'test_supabase_connection.py'.")
    print("Ensure 'test_supabase_connection.py' exists in the same directory and contains this function.")
    exit()

try:
    from db_operations import insert_toxicology_records, upsert_toxicology_records
except ImportError:
    print("ERROR: Could not import functions from 'db_operations.py'.")
    print("Ensure 'db_operations.py' exists in the same directory and contains these functions.")
    exit()


import logging
from datetime import datetime, timezone
import os # For a potential cleanup of temp PDF if that test was still here

# Configure logging (can be centralized in your main agent script eventually)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- IMPORTANT: Define your unique key strategy for upserting ---
# This MUST match the unique constraint you created in your Supabase table.
# Option 1: Using a list of column names that form the unique key
UNIQUE_KEY_COLUMNS_FOR_TEST = ["test_cas", "species_scientific_name", "endpoint", "conc1_mean", "conc1_unit", "source_url"]
# Option 2: Using the name of the constraint you created in SQL
# UNIQUE_CONSTRAINT_NAME_FOR_TEST = "unique_tox_record_key" # Replace with your actual constraint name

# --- Table Name Constants ---
TABLE_CHEMICALS = "toxicology_data"

def run_db_operations_test():
    logging.info("--- Starting Database Operations Test ---")
    supabase = init_supabase_client()
    if not supabase:
        logging.critical("Cannot proceed with DB operations test: Supabase client not initialized.")
        return

    # --- Test Data ---
    current_time_iso = datetime.now(timezone.utc).isoformat()
    record1_new = {
        "test_cas": "111-001-1", "chemical_name": "TestChem Alpha", # Made CAS slightly different
        "species_scientific_name": "Testus fishus", "species_common_name": "Test Fish",
        "species_group": "Fish", "endpoint": "LC50", "effect": "mortality",
        "conc1_mean": 10.5, "conc1_unit": "mg/L",
        "source_url": "http://example.com/test/alpha/unique1", "retrieval_date": current_time_iso,
        "validation_errors": [] # Assuming JSONB in DB, empty list is fine or set to None
    }
    record2_also_new = {
        "test_cas": "222-002-2", "chemical_name": "TestChem Beta", # Made CAS slightly different
        "species_scientific_name": "Daphnia testica", "species_common_name": "Test Daphnid",
        "species_group": "Invertebrate", "endpoint": "EC50", "effect": "immobilization",
        "conc1_mean": 1.23, "conc1_unit": "ug/L",
        "source_url": "http://example.com/test/beta/unique1", "retrieval_date": current_time_iso,
        "validation_errors": ["Minor observation noted"] # Example with validation error
    }
    # Record to test update (matches record1_new on unique key, but different effect and name)
    record1_updated_effect = {
        "test_cas": "111-001-1", "chemical_name": "TestChem Alpha (Updated Name)", # Name updated
        "species_scientific_name": "Testus fishus", "species_common_name": "Test Fish",
        "species_group": "Fish", "endpoint": "LC50", "effect": "SUBLETHAL STRESS", # Effect updated
        "conc1_mean": 10.5, "conc1_unit": "mg/L",
        "source_url": "http://example.com/test/alpha/unique1", "retrieval_date": datetime.now(timezone.utc).isoformat(), # Fresher date
        "validation_errors": None # Example: errors cleared on update
    }

    # --- Test 1: Basic Insert (using insert_toxicology_records) ---
    # This test is more meaningful if the records are guaranteed to be new and
    # you DON'T have the unique constraint yet, or if you want to test how
    # insert behaves when it HITS the unique constraint (it should fail).
    # For now, since we are focusing on upsert, we can comment this out to avoid
    # expected failures if the constraint is already active.
    # logging.info("\n--- Testing Basic Insert (MAY FAIL IF UNIQUE CONSTRAINT EXISTS AND RECORDS ARE DUPLICATES) ---")
    # test_inserts_for_basic_insert = [record1_new.copy(), record2_also_new.copy()] # Use fresh copies
    # inserted_results = insert_toxicology_records(supabase, test_inserts_for_basic_insert)
    # logging.info(f"Insert test: Supabase returned {len(inserted_results)} records.")
    # for res_idx, res_data in enumerate(inserted_results):
    #     logging.info(f"  Inserted {res_idx+1}: CAS {res_data.get('test_cas')} from {res_data.get('source_url')}")


    # --- Test 2: Upserting ---
    logging.info("\n--- Testing Upsert Operations ---")

    # Scenario A: Insert new records via upsert (first time these records are seen)
    logging.info("UPSERT SCENARIO A: Inserting record1_new and record2_also_new...")
    upsert_batch_A = [record1_new.copy(), record2_also_new.copy()] # Use fresh copies
    upserted_results_A = upsert_toxicology_records(
        supabase,
        upsert_batch_A,
        conflict_columns_list=UNIQUE_KEY_COLUMNS_FOR_TEST
        # OR use your named constraint: conflict_constraint_name=UNIQUE_CONSTRAINT_NAME_FOR_TEST
    )
    logging.info(f"Upsert Scenario A: Supabase processed {len(upserted_results_A)} records.")
    for res_idx, res_data in enumerate(upserted_results_A):
        logging.info(f"  Upserted A-{res_idx+1}: CAS {res_data.get('test_cas')}, Name: {res_data.get('chemical_name')}, Effect: {res_data.get('effect')}, Source: {res_data.get('source_url')}")

    # Scenario B: Update an existing record via upsert
    # record1_updated_effect should match record1_new on the unique key columns
    logging.info("\nUPSERT SCENARIO B: Updating existing record1 with record1_updated_effect...")
    upsert_batch_B = [record1_updated_effect.copy()] # Use a fresh copy
    upserted_results_B = upsert_toxicology_records(
        supabase,
        upsert_batch_B,
        conflict_columns_list=UNIQUE_KEY_COLUMNS_FOR_TEST
        # OR use your named constraint: conflict_constraint_name=UNIQUE_CONSTRAINT_NAME_FOR_TEST
    )
    logging.info(f"Upsert Scenario B: Supabase processed {len(upserted_results_B)} records.")
    if upserted_results_B:
        updated_record_data = upserted_results_B[0] # Expecting one record back
        logging.info(f"  Upserted B-1 (Updated): CAS {updated_record_data.get('test_cas')}, Name: {updated_record_data.get('chemical_name')}, Effect: {updated_record_data.get('effect')}")
        if updated_record_data.get('effect') == "SUBLETHAL STRESS" and \
           updated_record_data.get('chemical_name') == "TestChem Alpha (Updated Name)":
            logging.info("  SUCCESS: Record1 was correctly updated (effect and name changed).")
        else:
            logging.warning("  WARNING: Record1 update might not have reflected all changes as expected. Check DB and logs.")
    else:
        logging.error("  ERROR: Upsert Scenario B did not return any processed records.")

    # Scenario C: Attempt to insert what would be a duplicate if not for upsert
    # Using record1_new again, should just be an update (or no change if identical)
    logging.info("\nUPSERT SCENARIO C: Upserting record1_new again (should be an update to its current state)...")
    upsert_batch_C = [record1_new.copy()] # This is the original record1_new, but after record1_updated_effect was applied
                                         # So, this should effectively revert the effect/name if the DB still had the updated one.
                                         # Or, if we use the state from upserted_results_B[0], it would be no-op.
                                         # For a clear test, let's use the very first record1_new definition.
    original_record1_to_re_upsert = { # Re-define to ensure it's the original state
        "test_cas": "111-001-1", "chemical_name": "TestChem Alpha",
        "species_scientific_name": "Testus fishus", "species_common_name": "Test Fish",
        "species_group": "Fish", "endpoint": "LC50", "effect": "mortality",
        "conc1_mean": 10.5, "conc1_unit": "mg/L",
        "source_url": "http://example.com/test/alpha/unique1", "retrieval_date": current_time_iso, # Original retrieval date
        "validation_errors": []
    }
    upsert_batch_C = [original_record1_to_re_upsert.copy()]
    upserted_results_C = upsert_toxicology_records(
        supabase,
        upsert_batch_C,
        conflict_columns_list=UNIQUE_KEY_COLUMNS_FOR_TEST
    )
    logging.info(f"Upsert Scenario C: Supabase processed {len(upserted_results_C)} records.")
    if upserted_results_C:
        re_upserted_record_data = upserted_results_C[0]
        logging.info(f"  Upserted C-1 (Re-Upsert): CAS {re_upserted_record_data.get('test_cas')}, Name: {re_upserted_record_data.get('chemical_name')}, Effect: {re_upserted_record_data.get('effect')}")
        if re_upserted_record_data.get('effect') == "mortality" and \
           re_upserted_record_data.get('chemical_name') == "TestChem Alpha":
            logging.info("  SUCCESS: Record1 was correctly updated back to its original state (or remained if no intermediate changes were made by other tests).")
        else:
            logging.warning("  WARNING: Record1 re-upsert state is not as expected. Check DB. It should now reflect 'mortality' and 'TestChem Alpha'.")

    logging.info("--- Database Operations Test Finished ---")
    logging.info(f">>> Please check your Supabase '{TABLE_CHEMICALS}' table to verify results and your log file. <<<")
    logging.info(">>> Please check your Supabase 'toxicology_data' table to verify results and your log file. <<<")

if __name__ == "__main__":
    # Ensure .env file is loaded if init_supabase_client relies on python-dotenv
    # (The init_supabase_client from test_supabase_connection.py should handle this)
    run_db_operations_test()