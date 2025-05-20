# File: db_operations.py

from supabase import Client # For type hinting the supabase client object
import logging
import json # Will be used if 'validation_errors' column is TEXT

# It's good practice for this module to have its own logger
# or rely on the calling script's logger configuration.
# For now, assume logger is configured by the caller (e.g., test_db_ops.py or main_agent.py)
# logger = logging.getLogger(__name__) # Alternative if you want module-specific logger

# --- Table Name Constants ---
TABLE_CHEMICALS = "toxicology_data"

def insert_toxicology_records(supabase_client: Client, records: list[dict]) -> list[dict]:
    """
    Inserts a list of toxicology record dictionaries into the 'toxicology_data' table.
    Returns a list of records that Supabase confirmed were inserted/processed.
    """
    if not records:
        logging.info("db_operations: No records provided for insertion.")
        return []

    records_to_insert = []
    for record_idx, record_data in enumerate(records):
        db_record = record_data.copy()

        if 'needs_review' in db_record:
            del db_record['needs_review']

        if 'validation_errors' in db_record:
            if isinstance(db_record['validation_errors'], list):
                if not db_record['validation_errors']:
                    db_record['validation_errors'] = None
                # else if your column is TEXT:
                    # logging.debug(f"Record {record_idx}: Converting validation_errors list to JSON string.")
                    # db_record['validation_errors'] = json.dumps(db_record['validation_errors'])
            elif db_record['validation_errors'] is not None:
                logging.warning(f"db_operations: Record {record_idx} (source: {db_record.get('source_url')}): "
                                f"validation_errors is not a list ({type(db_record['validation_errors'])}). "
                                f"Converting to JSON array of strings: {db_record['validation_errors']}")
                try:
                    db_record['validation_errors'] = json.dumps([str(db_record['validation_errors'])]) # For TEXT
                    # If JSONB: db_record['validation_errors'] = [str(db_record['validation_errors'])]
                except Exception as e_json:
                    logging.error(f"db_operations: Error converting non-list validation_errors for record {record_idx}: {e_json}")
                    db_record['validation_errors'] = None

        if 'conc1_mean' in db_record and db_record['conc1_mean'] is not None:
            try:
                db_record['conc1_mean'] = float(db_record['conc1_mean'])
            except (ValueError, TypeError):
                logging.warning(f"db_operations: Record {record_idx} (source: {db_record.get('source_url')}): "
                                f"conc1_mean '{db_record['conc1_mean']}' could not be converted to float. Setting to None.")
                db_record['conc1_mean'] = None
        
        records_to_insert.append(db_record)
    
    if not records_to_insert:
        logging.info("db_operations: No suitable records to insert after pre-processing.")
        return []

    try:
        logging.info(f"db_operations: Attempting to batch insert {len(records_to_insert)} records into 'toxicology_data'.")
        response = supabase_client.table("toxicology_data").insert(records_to_insert).execute()

        if hasattr(response, 'error') and response.error:
            logging.error(f"db_operations: Supabase API Error during batch insert: {response.error.message} "
                          f"(Code: {response.error.code}, Details: {response.error.details})")
            return []
        elif hasattr(response, 'data'):
            logging.info(f"db_operations: Successfully executed batch insert. Supabase processed {len(response.data)} records.")
            return response.data
        else:
            logging.warning(f"db_operations: Batch insert executed, but response structure was unexpected. Response: {response}")
            return []

    except Exception as e:
        logging.error(f"db_operations: An exception occurred during Supabase batch insert: {e}", exc_info=True)
        return []


# --- Table Name Constants ---
TABLE_CHEMICALS = "toxicology_data"

def upsert_toxicology_records(supabase_client: Client, records: list[dict],
                             conflict_constraint_name: str | None = None,
                             conflict_columns_list: list[str] | None = None
                             ) -> list[dict]:
    """
    Upserts a list of toxicology record dictionaries into the 'toxicology_data' table.
    If a conflict occurs on the specified constraint or columns, the existing record is updated.
    Returns a list of records that Supabase confirmed were upserted/processed.
    """
    if not records:
        logging.info("db_operations: No records provided for upsertion.")
        return []

    if not conflict_constraint_name and not conflict_columns_list:
        logging.error("db_operations: CRITICAL: Upsert requires either a 'conflict_constraint_name' or 'conflict_columns_list'.")
        return []

    on_conflict_target = conflict_constraint_name
    if not on_conflict_target and conflict_columns_list:
         on_conflict_target = ",".join(conflict_columns_list)
    
    if not on_conflict_target:
        logging.error("db_operations: CRITICAL: No valid on_conflict target determined for upsert.")
        return []
        
    logging.info(f"db_operations: Using on_conflict target: '{on_conflict_target}' for upsert operation.")

    records_to_upsert = []
    for record_idx, record_data in enumerate(records):
        db_record = record_data.copy()
        if 'needs_review' in db_record:
            del db_record['needs_review']
        
        if 'validation_errors' in db_record:
            if isinstance(db_record['validation_errors'], list):
                if not db_record['validation_errors']:
                    db_record['validation_errors'] = None
                # else if column is TEXT:
                    # db_record['validation_errors'] = json.dumps(db_record['validation_errors'])
            elif db_record['validation_errors'] is not None:
                logging.warning(f"db_operations: Record {record_idx} (source: {db_record.get('source_url')}): "
                                f"validation_errors is not a list ({type(db_record['validation_errors'])}). "
                                f"Converting to JSON array of strings: {db_record['validation_errors']}")
                try:
                    db_record['validation_errors'] = json.dumps([str(db_record['validation_errors'])]) # For TEXT
                    # If JSONB: db_record['validation_errors'] = [str(db_record['validation_errors'])]
                except Exception as e_json:
                    logging.error(f"db_operations: Error converting non-list validation_errors for record {record_idx}: {e_json}")
                    db_record['validation_errors'] = None

        if 'conc1_mean' in db_record and db_record['conc1_mean'] is not None:
            try:
                val = float(db_record['conc1_mean'])
                # If conc1_mean is part of conflict_columns_list, consistent rounding might be needed for matching
                # if conflict_columns_list and 'conc1_mean' in conflict_columns_list:
                #     val = round(val, 5) # Example rounding
                db_record['conc1_mean'] = val
            except (ValueError, TypeError):
                logging.warning(f"db_operations: Record {record_idx} (source: {db_record.get('source_url')}): "
                                f"conc1_mean '{db_record['conc1_mean']}' could not be converted to float. Setting to None.")
                db_record['conc1_mean'] = None
        
        records_to_upsert.append(db_record)
    
    if not records_to_upsert:
        logging.info("db_operations: No suitable records to upsert after pre-processing.")
        return []

    try:
        logging.info(f"db_operations: Attempting to batch upsert {len(records_to_upsert)} records into 'toxicology_data'.")
        response = supabase_client.table("toxicology_data").upsert(
            records_to_upsert,
            on_conflict=on_conflict_target
        ).execute()

        if hasattr(response, 'error') and response.error:
            logging.error(f"db_operations: Supabase API Error during batch upsert: {response.error.message} "
                          f"(Code: {response.error.code}, Details: {response.error.details})")
            return []
        elif hasattr(response, 'data'):
            logging.info(f"db_operations: Successfully executed batch upsert. Supabase processed {len(response.data)} records.")
            return response.data
        else:
            logging.warning(f"db_operations: Batch upsert executed, but response structure was unexpected. Response: {response}")
            return []

    except Exception as e:
        logging.error(f"db_operations: An exception occurred during Supabase batch upsert: {e}", exc_info=True)
        return []