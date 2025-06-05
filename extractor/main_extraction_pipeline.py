# src/main_extraction_pipeline.py
import argparse
import logging
import json
import os
from typing import Dict, List, Any
from extractor.utils.logging_config import setup_logging
from extractor.utils.llm_client_setup import get_openai_client
from extractor.utils.file_operations import load_json_data, save_json_data
from extractor.utils.general_utils import PromptManager
from extractor.config.domain_specific_configs import get_domain_config
from extractor.config.prompt_templates import load_prompts as load_all_extraction_prompts
from extractor.extraction.core_processor import PaperProcessor  # PaperProcessor will take a new arg

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_and_log_stats(all_paper_stats: List[Dict[str, Any]], stats_output_path: str):
    # ... (Your existing calculate_and_log_stats function remains here) ...
    # This function will now report "SKIPPED" for Pydantic steps if validation was disabled.
    if not all_paper_stats:
        logger.info("No statistics collected to calculate rates.")
        return

    total_papers_attempted = len(all_paper_stats)
    total_parseable_llm_responses = 0
    total_initial_entries_from_llm = 0

    papers_passing_top_level_validation = 0
    papers_with_top_level_skipped = 0
    total_entries_after_top_level_validation = 0

    total_entries_attempted_domain_details_validation = 0
    total_entries_passing_domain_details_validation = 0
    entries_with_domain_details_skipped = 0

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
            # Count entries where domain validation was skipped (because top-level passed, but domain was skipped)
            for entry_stat in stats.get("domain_specific_validation_results", []):
                if entry_stat.get("passed") == "SKIPPED":
                    entries_with_domain_details_skipped += 1

        elif top_level_status == "SKIPPED":
            papers_with_top_level_skipped += 1
            # If top-level is skipped, domain-specific is also skipped. Count all entries as "skipped" for domain.
            num_entries_this_paper = stats.get("num_entries_after_top_level_validation",
                                               0)  # Should be initial_entry_count if top-level skipped
            total_entries_after_top_level_validation += num_entries_this_paper  # These didn't fail, they were skipped
            total_entries_attempted_domain_details_validation += num_entries_this_paper
            # In skipped mode, count_entries_passing_domain_details_validation would count all as "passed" by skipping
            total_entries_passing_domain_details_validation += stats.get(
                "count_entries_passing_domain_details_validation", 0)
            entries_with_domain_details_skipped += num_entries_this_paper

    # Calculate Rates
    json_parse_rate = (
                total_parseable_llm_responses / total_papers_attempted * 100) if total_papers_attempted > 0 else 0

    # Top-Level Schema Validation Rate (only for those not skipped)
    parseable_and_not_skipped_top_level = total_parseable_llm_responses - papers_with_top_level_skipped
    top_level_schema_validation_rate_per_paper = (
                papers_passing_top_level_validation / parseable_and_not_skipped_top_level * 100) if parseable_and_not_skipped_top_level > 0 else 0

    entry_survival_after_top_level_rate = (
                total_entries_after_top_level_validation / total_initial_entries_from_llm * 100) if total_initial_entries_from_llm > 0 else 0

    # Domain-Specific Details Validation Rate (only for those not skipped)
    attempted_and_not_skipped_domain = total_entries_attempted_domain_details_validation - entries_with_domain_details_skipped
    domain_specific_details_validation_rate_per_entry = (
                total_entries_passing_domain_details_validation / attempted_and_not_skipped_domain * 100) if attempted_and_not_skipped_domain > 0 else 0
    if papers_with_top_level_skipped == total_parseable_llm_responses and total_parseable_llm_responses > 0:  # All skipped
        domain_specific_details_validation_rate_per_entry = "SKIPPED"

    # End-to-End Validation Success Rate
    # If validation is skipped, this reflects entries that didn't fail basic parsing and made it to the end.
    # If validation is enabled, this reflects entries that passed all Pydantic checks.
    end_to_end_entry_success_rate = (
                total_entries_passing_domain_details_validation / total_initial_entries_from_llm * 100) if total_initial_entries_from_llm > 0 else 0

    logger.info("--- Pydantic Validation Statistics ---")
    logger.info(f"Total Papers Attempted: {total_papers_attempted}")
    logger.info(f"LLM Responses Parseable as JSON: {total_parseable_llm_responses} ({json_parse_rate:.2f}%)")
    if papers_with_top_level_skipped > 0:
        logger.info(f"Papers where Pydantic Validation was SKIPPED: {papers_with_top_level_skipped}")

    if parseable_and_not_skipped_top_level > 0:
        logger.info(
            f"Papers Passing Top-Level Pydantic Schema Validation (LLMOutputSchema): {papers_passing_top_level_validation} out of {parseable_and_not_skipped_top_level} attempted ({top_level_schema_validation_rate_per_paper:.2f}%)")
    else:
        logger.info("Top-Level Pydantic Schema Validation was not performed on any papers (or all were skipped).")

    logger.info(f"Total Initial Material Entries from LLM (sum over all papers): {total_initial_entries_from_llm}")
    logger.info(f"Total Entries After Top-Level Validation/Processing: {total_entries_after_top_level_validation}")
    logger.info(
        f"  - Entry Survival Rate after Top-Level Processing (vs Initial LLM Entries): {entry_survival_after_top_level_rate:.2f}%")

    if attempted_and_not_skipped_domain > 0:
        logger.info(
            f"Total Entries Attempted for Domain-Specific Details Validation (and not skipped): {attempted_and_not_skipped_domain}")
        logger.info(
            f"Entries Passing Domain-Specific Details Pydantic Validation: {total_entries_passing_domain_details_validation} ({domain_specific_details_validation_rate_per_entry:.2f}%)")
    elif isinstance(domain_specific_details_validation_rate_per_entry,
                    str) and domain_specific_details_validation_rate_per_entry == "SKIPPED":
        logger.info("Domain-Specific Details Pydantic Validation was SKIPPED for all entries.")
    else:
        logger.info(
            "Domain-Specific Details Pydantic Validation was not performed on any entries (or all were skipped).")

    logger.info(
        f"End-to-End Entry Success Rate (vs Initial LLM Entries): {end_to_end_entry_success_rate:.2f}% (Note: if validation skipped, this reflects entries passing basic parsing)")
    logger.info("------------------------------------")
    # ... (rest of saving detailed_stats_to_save as before) ...


def main():
    parser = argparse.ArgumentParser(description="Run the academic paper information extraction pipeline.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSON file containing paper data (output from PDF processor).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the extracted structured information (final valid data).")
    parser.add_argument("--stats_file", type=str, default="data/output/extraction_validation_stats.json",
                        help="Path to save the detailed Pydantic validation statistics.")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint_extraction.json",
                        help="Path for the checkpoint file.")
    parser.add_argument("--domain", type=str, default="in2o3_tco",
                        help="The domain for extraction (e.g., 'membrane', 'in2o3_tco').")
    parser.add_argument("--language", type=str, default="en", help="Language of the papers and prompts (e.g., 'en').")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-671B",
                        help="Name of the LLM to use for extraction.")
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--disable_pydantic_validation",
        action="store_true",  # If flag is present, set to True
        help="If set, Pydantic schema validation will be skipped."
    )

    args = parser.parse_args()
    logger.info(f"Starting extraction pipeline with args: {args}")

    # Determine if Pydantic validation should be enabled
    pydantic_enabled = not args.disable_pydantic_validation  # Enable if flag is NOT set
    if not pydantic_enabled:
        logger.warning("PYDANTIC VALIDATION IS DISABLED VIA COMMAND-LINE ARGUMENT.")

    try:
        openai_client = get_openai_client()
        domain_config = get_domain_config(args.domain)
        logger.info(f"Loaded configuration for domain: {args.domain}")
        prompt_manager = PromptManager()
        load_all_extraction_prompts(prompt_manager)
        logger.info(f"Prompts loaded. Available for: {list(prompt_manager.templates.keys())}")

        processor = PaperProcessor(
            client=openai_client,
            prompt_manager=prompt_manager,
            domain_config=domain_config,
            model_name=args.model_name,
            pydantic_validation_enabled=pydantic_enabled  # <--- PASS THE FLAG
        )
        logger.info(
            f"PaperProcessor initialized with model: {args.model_name} and Pydantic validation {'enabled' if pydantic_enabled else 'disabled'}.")

        papers_data_list = load_json_data(args.input_file)
        if not isinstance(papers_data_list, list):
            logger.error(f"Input file {args.input_file} does not contain a list.")
            return
        logger.info(f"Loaded {len(papers_data_list)} paper data items from {args.input_file}")

        extracted_results, all_paper_stats = processor.process_papers_with_checkpoint(
            papers_list=papers_data_list,
            domain_name=args.domain,
            language=args.language,
            checkpoint_file_path=args.checkpoint_file
            # pydantic_enabled is now part of the processor instance
        )

        save_json_data(extracted_results, args.output_file)
        logger.info(f"Extraction complete. {len(extracted_results)} material entries saved to {args.output_file}")

        calculate_and_log_stats(all_paper_stats, args.stats_file)

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
    except FileNotFoundError as fnfe:
        logger.error(f"File access error: {fnfe}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()