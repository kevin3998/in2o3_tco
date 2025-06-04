# src/main_extraction_pipeline.py
import argparse
import logging
import json # For saving stats
import os # For path operations
from typing import List, Dict, Any

from extractor.utils.logging_config import setup_logging
from extractor.utils.llm_client_setup import get_openai_client
from extractor.utils.file_operations import load_json_data, save_json_data # save_json_data for main results
from extractor.utils.general_utils import PromptManager
from extractor.config.domain_specific_configs import get_domain_config
from extractor.config.prompt_templates import load_prompts as load_all_extraction_prompts
from extractor.extraction.core_processor import PaperProcessor

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_and_log_stats(all_paper_stats: List[Dict[str, Any]], stats_output_path: str):
    if not all_paper_stats:
        logger.info("No statistics collected to calculate rates.")
        return

    total_papers_attempted = len(all_paper_stats)
    total_parseable_llm_responses = 0
    total_initial_entries_from_llm = 0

    papers_passing_top_level_validation = 0
    total_entries_after_top_level_validation = 0 # Sum of num_entries_after_top_level_validation from each paper

    total_entries_attempted_domain_details_validation = 0 # This is total_entries_after_top_level_validation
    total_entries_passing_domain_details_validation = 0

    for stats in all_paper_stats:
        if stats.get("raw_llm_response_parseable_as_json", False):
            total_parseable_llm_responses += 1

        total_initial_entries_from_llm += stats.get("initial_entry_count_from_llm", 0)

        if stats.get("top_level_validation_passed", False):
            papers_passing_top_level_validation += 1
            num_entries_this_paper = stats.get("num_entries_after_top_level_validation", 0)
            total_entries_after_top_level_validation += num_entries_this_paper
            total_entries_attempted_domain_details_validation += num_entries_this_paper # These are the ones for which domain validation is tried
            total_entries_passing_domain_details_validation += stats.get("count_entries_passing_domain_details_validation", 0)

    # Calculate Rates
    json_parse_rate = (total_parseable_llm_responses / total_papers_attempted * 100) if total_papers_attempted > 0 else 0
    top_level_schema_validation_rate_per_paper = (papers_passing_top_level_validation / total_parseable_llm_responses * 100) if total_parseable_llm_responses > 0 else 0

    # Entry-Level Schema Validation Rate:
    # This is implicitly 100% for entries that are part of a top-level validated output.
    # So, a more meaningful metric here is the proportion of initial LLM entries that made it past top-level validation.
    entry_survival_after_top_level_rate = (total_entries_after_top_level_validation / total_initial_entries_from_llm * 100) if total_initial_entries_from_llm > 0 else 0

    domain_specific_details_validation_rate_per_entry = (total_entries_passing_domain_details_validation / total_entries_attempted_domain_details_validation * 100) if total_entries_attempted_domain_details_validation > 0 else 0

    # End-to-End Validation Success Rate (per entry initially proposed by LLM)
    end_to_end_entry_success_rate = (total_entries_passing_domain_details_validation / total_initial_entries_from_llm * 100) if total_initial_entries_from_llm > 0 else 0

    logger.info("--- Pydantic Validation Statistics ---")
    logger.info(f"Total Papers Attempted: {total_papers_attempted}")
    logger.info(f"LLM Responses Parseable as JSON: {total_parseable_llm_responses} ({json_parse_rate:.2f}%)")
    logger.info(f"Total Initial Material Entries from LLM (sum over all papers): {total_initial_entries_from_llm}")
    logger.info(f"Papers Passing Top-Level Pydantic Schema Validation (LLMOutputSchema): {papers_passing_top_level_validation} out of {total_parseable_llm_responses} parseable responses ({top_level_schema_validation_rate_per_paper:.2f}%)")
    logger.info(f"Total Entries After Top-Level Validation (structurally valid ExtractedMaterialEntrySchema): {total_entries_after_top_level_validation}")
    logger.info(f"  - Implied Entry Structural Validation Rate (ExtractedMaterialEntrySchema vs Initial LLM Entries): {entry_survival_after_top_level_rate:.2f}%")
    logger.info(f"Total Entries Attempted for Domain-Specific Details Validation: {total_entries_attempted_domain_details_validation}")
    logger.info(f"Entries Passing Domain-Specific Details Pydantic Validation: {total_entries_passing_domain_details_validation} ({domain_specific_details_validation_rate_per_entry:.2f}%)")
    logger.info(f"End-to-End Entry Validation Success Rate (vs Initial LLM Entries): {end_to_end_entry_success_rate:.2f}%")
    logger.info("------------------------------------")

    # Save detailed stats
    detailed_stats_to_save = {
        "summary_rates": {
            "total_papers_attempted": total_papers_attempted,
            "json_parse_rate_percent": json_parse_rate,
            "top_level_schema_validation_rate_per_parseable_paper_percent": top_level_schema_validation_rate_per_paper,
            "entry_structural_validation_rate_vs_initial_llm_entries_percent": entry_survival_after_top_level_rate,
            "domain_specific_details_validation_rate_per_entry_percent": domain_specific_details_validation_rate_per_entry,
            "end_to_end_entry_success_rate_percent": end_to_end_entry_success_rate,
            "total_initial_entries_from_llm": total_initial_entries_from_llm,
            "total_entries_passing_all_validation": total_entries_passing_domain_details_validation
        },
        "per_paper_details": all_paper_stats
    }
    try:
        output_dir = os.path.dirname(stats_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(stats_output_path, "w", encoding="utf-8") as f_stats:
            json.dump(detailed_stats_to_save, f_stats, indent=2, ensure_ascii=False)
        logger.info(f"Detailed validation statistics saved to: {stats_output_path}")
    except Exception as e:
        logger.error(f"Failed to save detailed validation statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run the academic paper information extraction pipeline.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing paper data (output from PDF processor).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the extracted structured information (final valid data).")
    parser.add_argument("--stats_file", type=str, default="data/extracted_json/extraction_validation_stats.json", help="Path to save the detailed Pydantic validation statistics.")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint_extraction.json", help="Path for the checkpoint file.")
    parser.add_argument("--domain", type=str, default="in2o3_tco", help="The domain for extraction (e.g., 'membrane', 'in2o3_tco').") # Default to TCO
    parser.add_argument("--language", type=str, default="en", help="Language of the papers and prompts (e.g., 'en').")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-671B", help="Name of the LLM to use for extraction.")

    args = parser.parse_args()
    logger.info(f"Starting extraction pipeline with args: {args}")

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
            model_name=args.model_name
        )
        logger.info(f"PaperProcessor initialized with model: {args.model_name}")

        papers_data_list = load_json_data(args.input_file)
        if not isinstance(papers_data_list, list):
            logger.error(f"Input file {args.input_file} does not contain a list.")
            return
        logger.info(f"Loaded {len(papers_data_list)} paper data items from {args.input_file}")

        # process_papers_with_checkpoint now returns (extracted_results, all_paper_stats)
        extracted_results, all_paper_stats = processor.process_papers_with_checkpoint(
            papers_list=papers_data_list,
            domain_name=args.domain,
            language=args.language,
            checkpoint_file_path=args.checkpoint_file
        )

        save_json_data(extracted_results, args.output_file) # Save the successfully extracted data
        logger.info(f"Extraction complete. {len(extracted_results)} material entries saved to {args.output_file}")

        # Calculate and log/save validation statistics
        calculate_and_log_stats(all_paper_stats, args.stats_file)

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
    except FileNotFoundError as fnfe:
        logger.error(f"File access error: {fnfe}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()