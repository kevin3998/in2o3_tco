"""
Module: main_extraction_pipeline
Functionality: Serves as the main entry point for running the information
               extraction pipeline. It handles command-line argument parsing,
               initializes necessary components (LLM client, configurations,
               processor), loads input data, orchestrates the processing,
               and saves the final results and statistics.
"""
import argparse
import logging
import json
import os
import traceback
from typing import List, Dict, Any
from extractor.utils.logging_config import setup_logging
from extractor.utils.llm_client_setup import get_openai_client
from extractor.utils.file_operations import load_json_data, save_json_data, cleanup_checkpoint
from extractor.utils.general_utils import PromptManager
from extractor.config.domain_specific_configs import get_domain_config
from extractor.config.prompt_templates import load_prompts as load_all_extraction_prompts
from extractor.extraction.core_processor import PaperProcessor

# Setup logging as early as possible
setup_logging(level=logging.INFO)  # Set to logging.DEBUG for more verbosity
logger = logging.getLogger(__name__)


def calculate_and_log_stats(all_paper_stats: List[Dict[str, Any]], stats_output_path: str):
    """
    Calculates and logs detailed validation statistics, then saves them to a file.
    """
    if not all_paper_stats:
        logger.info("No statistics collected to calculate rates. Stats file will not be created.")
        return

    # --- Initialize Counters ---
    total_papers_attempted = len(all_paper_stats)
    total_parseable_llm_responses = 0
    total_initial_entries_from_llm = 0

    papers_passing_top_level_validation = 0
    papers_with_top_level_skipped = 0
    total_entries_after_top_level_validation = 0

    total_entries_attempted_domain_details_validation = 0
    total_entries_passing_domain_details_validation = 0
    entries_with_domain_details_skipped = 0

    # --- Aggregate Data ---
    for stats in all_paper_stats:
        if stats.get("raw_llm_response_parseable_as_json", False):
            total_parseable_llm_responses += 1

        total_initial_entries_from_llm += stats.get("initial_entry_count_from_llm", 0)

        top_level_status = stats.get("top_level_validation_passed", False)
        if top_level_status is True:
            papers_passing_top_level_validation += 1
            num_entries_this_paper = stats.get("num_entries_after_top_level_validation", 0)
            total_entries_after_top_level_validation += num_entries_this_paper
            total_entries_attempted_domain_details_validation += num_entries_this_paper
            total_entries_passing_domain_details_validation += stats.get(
                "count_entries_passing_domain_details_validation", 0)
        elif top_level_status == "SKIPPED":
            papers_with_top_level_skipped += 1
            num_entries_this_paper = stats.get("num_entries_after_top_level_validation", 0)
            total_entries_after_top_level_validation += num_entries_this_paper
            # In skipped mode, assume all entries would be attempted for domain validation
            total_entries_attempted_domain_details_validation += num_entries_this_paper
            # In skipped mode, all are counted as "passing" by bypassing validation
            total_entries_passing_domain_details_validation += stats.get(
                "count_entries_passing_domain_details_validation", 0)
            entries_with_domain_details_skipped += num_entries_this_paper

    # --- Calculate Rates ---
    json_parse_rate = (
            total_parseable_llm_responses / total_papers_attempted * 100) if total_papers_attempted > 0 else 0

    parseable_and_not_skipped_top_level = total_parseable_llm_responses - papers_with_top_level_skipped
    top_level_schema_validation_rate_per_paper = (
            papers_passing_top_level_validation / parseable_and_not_skipped_top_level * 100) if parseable_and_not_skipped_top_level > 0 else 0

    entry_survival_after_top_level_rate = (
            total_entries_after_top_level_validation / total_initial_entries_from_llm * 100) if total_initial_entries_from_llm > 0 else 0

    attempted_and_not_skipped_domain = total_entries_attempted_domain_details_validation - entries_with_domain_details_skipped
    domain_specific_details_validation_rate_per_entry = (
            total_entries_passing_domain_details_validation / attempted_and_not_skipped_domain * 100) if attempted_and_not_skipped_domain > 0 else "N/A"

    if papers_with_top_level_skipped == total_parseable_llm_responses and total_parseable_llm_responses > 0:  # All skipped
        domain_specific_details_validation_rate_per_entry = "SKIPPED"

    end_to_end_entry_success_rate = (
            total_entries_passing_domain_details_validation / total_initial_entries_from_llm * 100) if total_initial_entries_from_llm > 0 else 0

    # --- Log Summary to Console ---
    logger.info("--- Pydantic Validation Statistics ---")
    logger.info(f"Total Papers Attempted: {total_papers_attempted}")
    logger.info(f"LLM Responses Parseable as JSON: {total_parseable_llm_responses} ({json_parse_rate:.2f}%)")
    if papers_with_top_level_skipped > 0:
        logger.info(f"Papers where Pydantic Validation was SKIPPED: {papers_with_top_level_skipped}")

    if parseable_and_not_skipped_top_level > 0:
        logger.info(
            f"Papers Passing Top-Level Schema Validation: {papers_passing_top_level_validation} / {parseable_and_not_skipped_top_level} attempted ({top_level_schema_validation_rate_per_paper:.2f}%)")
    else:
        logger.info("Top-Level Schema Validation was not performed on any papers (or all were skipped).")

    logger.info(f"Total Initial Material Entries from LLM: {total_initial_entries_from_llm}")
    logger.info(
        f"Total Entries Surviving Top-Level Processing: {total_entries_after_top_level_validation} ({entry_survival_after_top_level_rate:.2f}% of initial)")

    if attempted_and_not_skipped_domain > 0:
        logger.info(
            f"Entries Passing Domain-Specific Details Validation: {total_entries_passing_domain_details_validation} / {attempted_and_not_skipped_domain} attempted ({domain_specific_details_validation_rate_per_entry:.2f}%)")
    elif isinstance(domain_specific_details_validation_rate_per_entry,
                    str) and domain_specific_details_validation_rate_per_entry == "SKIPPED":
        logger.info("Domain-Specific Details Pydantic Validation was SKIPPED for all entries.")
    else:
        logger.info("Domain-Specific Details Validation was not performed on any entries (or all were skipped).")

    logger.info(f"End-to-End Entry Success Rate (vs Initial): {end_to_end_entry_success_rate:.2f}%")
    logger.info("------------------------------------")

    # --- Save Detailed Stats to File ---
    logger.info("Attempting to save detailed validation statistics...")

    detailed_stats_to_save = {
        "summary_rates": {
            "total_papers_attempted": total_papers_attempted,
            "json_parse_rate_percent": json_parse_rate,
            "top_level_schema_validation_rate_per_parseable_paper_percent": top_level_schema_validation_rate_per_paper,
            "entry_survival_after_top_level_processing_percent": entry_survival_after_top_level_rate,
            "domain_specific_details_validation_rate_per_entry_percent": domain_specific_details_validation_rate_per_entry,
            "end_to_end_entry_success_rate_percent": end_to_end_entry_success_rate,
            "total_initial_entries_from_llm": total_initial_entries_from_llm,
            "total_fully_validated_entries": total_entries_passing_domain_details_validation
        },
        "per_paper_details": all_paper_stats
    }

    try:
        output_dir = os.path.dirname(stats_output_path)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory for stats file: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Writing stats to file: {stats_output_path}")
        with open(stats_output_path, "w", encoding="utf-8") as f_stats:
            json.dump(detailed_stats_to_save, f_stats, indent=2, ensure_ascii=False)
        logger.info(f"✅ Detailed validation statistics saved successfully to: {stats_output_path}")
    except TypeError as te:
        logger.error(f"❌ FAILED TO SAVE STATS FILE due to a TypeError (object not JSON serializable).")
        logger.error(f"Error Message: {te}")
        # Re-raise the exception to ensure the main try-except block catches it
        # and prevents the checkpoint from being deleted.
        raise
    except Exception as e:
        logger.error(f"❌ FAILED TO SAVE STATS FILE due to an unexpected exception.")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        # Re-raise the exception
        raise


def main():
    # =================================================================================
    # === Configuration Block for PyCharm/IDE Run ===
    # Set to True to use the settings below.
    # Set to False to use command-line arguments when running from the terminal.
    IDE_RUN_CONFIG = True
    # =================================================================================

    args = None
    if IDE_RUN_CONFIG:
        logger.info("--- RUNNING IN IDE MODE: Using parameters defined in the script. ---")
        # Use a simple class to mimic the argparse.Namespace object
        class Args:
            pass
        args = Args()

        # --- SET YOUR PARAMETERS FOR IDE RUNS HERE ---
        args.input_file = "../data/processed_text/processed_papers.json"
        args.output_file = "../data/extracted_json/pydantic/structured_info.json"
        args.stats_file = "../data/extracted_json/pydantic/stats.json"
        args.checkpoint_file = "checkpoint_extraction.json"
        args.domain = "in2o3_tco"
        args.language = "en"
        args.model_name = "DeepSeek-R1-671B"
        args.disable_pydantic_validation = False
        # ---------------------------------------------

    else:
        logger.info("--- RUNNING IN COMMAND-LINE MODE: Parsing arguments from terminal. ---")
        parser = argparse.ArgumentParser(description="Run the academic paper information extraction pipeline.")
        parser.add_argument("--input_file", type=str, required=True,
                            help="Path to the input JSON file containing paper data (output from PDF processor).")
        parser.add_argument("--output_file", type=str, required=True,
                            help="Path to save the extracted structured information (final valid data).")
        parser.add_argument("--stats_file", type=str, default="data/extraction_validation_stats.json",
                            help="Path to save the detailed Pydantic validation statistics.")
        parser.add_argument("--checkpoint_file", type=str, default="checkpoint_extraction.json",
                            help="Path for the checkpoint file.")
        parser.add_argument("--domain", type=str, default="in2o3_tco",
                            help="The domain for extraction (e.g., 'membrane', 'in2o3_tco').")
        parser.add_argument("--language", type=str, default="en", help="Language of the papers and prompts (e.g., 'en').")
        parser.add_argument("--model_name", type=str, default="DeepSeek-R1-671B",
                            help="Name of the LLM to use for extraction.")
        parser.add_argument(
            "--disable_pydantic_validation",
            action="store_true",
            help="If set, Pydantic schema validation will be skipped."
        )
        args = parser.parse_args()

        logger.info(f"Starting extraction pipeline with args: {vars(args)}")

        pydantic_enabled = not args.disable_pydantic_validation
        if not pydantic_enabled:
            logger.warning("PYDANTIC VALIDATION IS DISABLED.")

        try:
            # Step 1: Initialization
            openai_client = get_openai_client()
            domain_config = get_domain_config(args.domain)
            logger.info(f"Loaded configuration for domain: {args.domain}")
            prompt_manager = PromptManager()
            load_all_extraction_prompts(prompt_manager)
            logger.info(f"Prompts loaded. Available for: {list(prompt_manager.templates.keys())}")

            # Step 2: Create Processor
            processor = PaperProcessor(
                client=openai_client,
                prompt_manager=prompt_manager,
                domain_config=domain_config,
                model_name=args.model_name,
                pydantic_validation_enabled=pydantic_enabled
            )
            logger.info(
                f"PaperProcessor initialized with model: {args.model_name} and Pydantic validation {'enabled' if pydantic_enabled else 'disabled'}.")

            # Step 3: Load Data
            papers_data_list = load_json_data(args.input_file)
            if not isinstance(papers_data_list, list):
                logger.error(f"Input file {args.input_file} does not contain a list.")
                return
            logger.info(f"Loaded {len(papers_data_list)} paper data items from {args.input_file}")

            # Step 4: Run Extraction
            extracted_results, all_paper_stats = processor.process_papers_with_checkpoint(
                papers_list=papers_data_list,
                domain_name=args.domain,
                language=args.language,
                checkpoint_file_path=args.checkpoint_file
            )

            # Step 5: Save Main Output File
            save_json_data(extracted_results, args.output_file)
            logger.info(
                f"Extraction complete. {len(extracted_results)} valid material entries saved to {args.output_file}")

            # Step 6: Save Statistics File
            calculate_and_log_stats(all_paper_stats, args.stats_file)

            # --- MODIFICATION: CLEANUP CHECKPOINT AT THE VERY END ---
            # This code will only be reached if both save_json_data and calculate_and_log_stats complete successfully.
            # This prevents the data loss you experienced previously.
            logger.info("All data saved successfully. Cleaning up checkpoint file.")
            cleanup_checkpoint(args.checkpoint_file)
            # --- END OF MODIFICATION ---

        except Exception as e:
            logger.critical(
                f"An unexpected error occurred in the main pipeline and the process will terminate. The checkpoint file has been kept for recovery.")
            logger.critical(f"Error details: {e}", exc_info=True)
            # If any error occurs in the steps above (e.g., in save_json_data or calculate_and_log_stats),
            # the program jumps here, skipping the cleanup_checkpoint call.

    if __name__ == "__main__":
        main()