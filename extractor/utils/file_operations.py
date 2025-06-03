"""
Module: file_operations
Functionality: Provides utility functions for file input/output operations.
               This includes loading and saving JSON data, as well as managing
               checkpoint files for resuming long-running processes.
"""
import json
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Any:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        raise

def save_json_data(data: Any, file_path: str, indent: int = 2):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON data to {file_path}: {e}")
        raise

# src/utils/file_operations.py
def save_checkpoint(checkpoint_path: str, data_to_save: dict):
    try:
        # Data to be saved might have 'processed_ids' as a set
        data_copy = data_to_save.copy() # Work on a copy to avoid modifying the original dict in memory if it's reused

        # Standardize to 'processed_ids' for saving if old keys are present
        if 'processed_dois' in data_copy and isinstance(data_copy['processed_dois'], set):
            data_copy['processed_ids'] = list(data_copy.pop('processed_dois'))
        elif 'processed_ids' in data_copy and isinstance(data_copy['processed_ids'], set):
            data_copy['processed_ids'] = list(data_copy['processed_ids'])
        # else: assume it's already a list or not present, or an error if it's a set under a different key

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data_copy, f, indent=2, ensure_ascii=False)
        # No need to convert back to set here; loading logic should handle creating a set.
        logger.debug(f"Checkpoint saved to {checkpoint_path} with processed_ids as list.")
    except TypeError as te: # Specifically catch TypeError for non-serializable
        logger.error(f"TypeError saving checkpoint to {checkpoint_path}: {te}. Data causing issue (keys): {list(data_to_save.keys())}")
        # Log more details about what might be non-serializable
        for k, v in data_to_save.items():
            if isinstance(v, set):
                logger.error(f"Field '{k}' is a set and was not converted to list for checkpoint.")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")

# Also, update load_checkpoint to ensure 'processed_ids' is a set
def load_checkpoint(checkpoint_path: str, default_checkpoint: dict) -> dict:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            loaded_checkpoint = {**default_checkpoint, **checkpoint} # Merge with default

            # Ensure 'processed_ids' (or 'processed_dois' for backward compatibility) becomes a set
            processed_items_key = None
            if 'processed_ids' in loaded_checkpoint:
                processed_items_key = 'processed_ids'
            elif 'processed_dois' in loaded_checkpoint: # Check for old key
                processed_items_key = 'processed_dois'

            if processed_items_key and isinstance(loaded_checkpoint[processed_items_key], list):
                loaded_checkpoint['processed_ids'] = set(loaded_checkpoint.pop(processed_items_key)) # Standardize to 'processed_ids' and convert to set
            elif processed_items_key and not isinstance(loaded_checkpoint[processed_items_key], set):
                 loaded_checkpoint['processed_ids'] = set() # Initialize if not a list or set
            elif not processed_items_key: # If neither key exists
                loaded_checkpoint['processed_ids'] = set()


            logger.info(
                f"Checkpoint loaded from {checkpoint_path} | Processed: {loaded_checkpoint.get('total_processed', 0)} | Elapsed: {loaded_checkpoint.get('total_elapsed', 0.0):.1f}s"
            )
            return loaded_checkpoint
        except Exception as e:
            logger.warning(f"Checkpoint loading failed from {checkpoint_path}: {e}. Starting fresh.")
            default_checkpoint['processed_ids'] = set() # Ensure default also has the set
            return default_checkpoint
    logger.info("No checkpoint found. Starting fresh.")
    default_checkpoint['processed_ids'] = set() # Ensure default also has the set
    return default_checkpoint

def cleanup_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint file {os.path.basename(checkpoint_path)} cleaned up.")
        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed for {checkpoint_path}: {e}")