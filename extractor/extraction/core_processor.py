# src/extraction/core_processor.py
import time
from typing import List, Dict, Any, Tuple  # Added Tuple
from openai import OpenAI
from extractor.utils.general_utils import PromptManager
from extractor.config.domain_specific_configs import DomainConfig
from extractor.extraction.response_parser import parse_llm_response  # parse_llm_response now returns a tuple
from extractor.utils.file_operations import load_checkpoint, save_checkpoint, cleanup_checkpoint
import logging
from pathlib import Path  # Added Path for filename extraction

logger = logging.getLogger(__name__)


class PaperProcessor:
    def __init__(self, client: OpenAI, prompt_manager: PromptManager, domain_config: DomainConfig,
                 model_name: str = "gpt-4"):
        self.client = client
        self.prompt_manager = prompt_manager
        self.domain_config = domain_config
        self.model_name = model_name

    def _get_text_for_llm(self, paper_data: Dict) -> str:
        # ... (your existing _get_text_for_llm method, ensure it's robust) ...
        llm_text = paper_data.get("llm_ready_fulltext_cleaned", "")
        if isinstance(llm_text, str) and llm_text.strip():
            return llm_text.strip()
        abstract_text = paper_data.get("extracted_abstract_cleaned", "")
        if isinstance(abstract_text, str) and abstract_text.strip():
            logger.info(
                f"Using 'extracted_abstract_cleaned' for paper DOI: {paper_data.get('doi', 'N/A')} as 'llm_ready_fulltext_cleaned' is missing or empty.")
            return abstract_text.strip()
        logger.warning(
            f"No usable text found for paper DOI: {paper_data.get('doi', 'N/A')}, Title: {paper_data.get('retrieved_title', 'N/A')}. Available keys: {list(paper_data.keys())}. Returning empty string.")
        return ""

    def process_single_paper_llm_call(self, paper_data: Dict, domain_name: str, language: str) -> Tuple[
        List[Dict], Dict[str, Any]]:
        # Return type changed to Tuple[List[Dict], Dict[str, Any]]
        text_to_process = self._get_text_for_llm(paper_data)
        paper_id_for_log = paper_data.get("doi", Path(paper_data.get("local_path", "unknown")).name)

        if not text_to_process:
            logger.warning(f"Skipping paper {paper_id_for_log} due to empty text for LLM.")
            # Return empty results and basic stats indicating failure at this stage
            empty_stats = {
                "paper_id": paper_id_for_log, "raw_llm_response_parseable_as_json": False,
                "initial_entry_count_from_llm": 0, "top_level_validation_passed": False,
                "top_level_error_message": "No text provided to LLM.",
                "num_entries_after_top_level_validation": 0,
                "domain_specific_validation_results": [],
                "count_entries_passing_domain_details_validation": 0
            }
            return [], empty_stats

        prompt_for_llm = self.prompt_manager.get_prompt(domain_name, language, text_to_process)
        max_retries = 3

        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} for paper {paper_id_for_log}")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert academic extraction assistant specialized in materials science. Adhere strictly to the user's requested JSON output format."},
                        {"role": "user", "content": prompt_for_llm}
                    ],
                    temperature=0.1, stream=False
                )
                llm_response_content = response.choices[0].message.content

                # parse_llm_response now returns (list_of_entries, paper_stats_dict)
                return parse_llm_response(
                    llm_content=llm_response_content,
                    original_input_text_for_llm=text_to_process,
                    paper_meta=paper_data,
                    domain_config=self.domain_config,
                    language=language
                )
            except Exception as e:
                logger.error(f"LLM call or parsing attempt {attempt + 1} failed for {paper_id_for_log}: {e}")
                if attempt < max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Final attempt failed for {paper_id_for_log}. Skipping this paper.")
                    # Return empty results and stats indicating LLM call failure
                    fail_stats = {
                        "paper_id": paper_id_for_log, "raw_llm_response_parseable_as_json": False,
                        "initial_entry_count_from_llm": 0, "top_level_validation_passed": False,
                        "top_level_error_message": f"LLM call failed after {max_retries} retries: {str(e)}",
                        "num_entries_after_top_level_validation": 0,
                        "domain_specific_validation_results": [],
                        "count_entries_passing_domain_details_validation": 0
                    }
                    return [], fail_stats
        # Fallback if loop finishes unexpectedly (e.g. max_retries = 0)
        unreachable_stats = {"paper_id": paper_id_for_log,
                             "top_level_error_message": "Max retries was zero or loop ended."}  # Simplified
        return [], unreachable_stats

    def process_papers_with_checkpoint(self,
                                       papers_list: List[Dict],
                                       domain_name: str,
                                       language: str,
                                       checkpoint_file_path: str = "checkpoint_extraction.json"
                                       ) -> Tuple[List[Dict], List[Dict[str, Any]]]:  # Return type changed
        # ... (checkpoint loading logic - ensure 'processed_ids' is a set) ...
        # ... (variable initializations) ...
        all_extracted_results = []  # This will store List[Dict] of successfully extracted material_entries
        all_paper_stats = []  # This will store List[Dict[str, Any]] of stats for each paper

        # Load checkpoint
        default_checkpoint_data = {"processed_ids": set(), "results": [], "all_paper_stats": [], "total_elapsed": 0.0,
                                   "total_processed": 0}
        checkpoint_data = load_checkpoint(checkpoint_file_path, default_checkpoint_data)

        processed_ids = checkpoint_data.get("processed_ids", set())  # Ensure it's a set
        all_extracted_results = checkpoint_data.get("results", [])
        all_paper_stats = checkpoint_data.get("all_paper_stats", [])  # Load previous stats
        total_elapsed_time = checkpoint_data.get("total_elapsed", 0.0)
        num_already_processed_papers = checkpoint_data.get("total_processed", 0)

        # ... (rest of the setup) ...
        total_papers_to_process = len(papers_list)
        logger.info(
            f"Starting batch processing for structured extraction. Total input items: {total_papers_to_process}. Already processed from checkpoint: {num_already_processed_papers}.")
        newly_processed_paper_count_this_session = 0

        try:
            for idx, paper_data_item in enumerate(papers_list):
                paper_id = paper_data_item.get("doi", None)
                if not paper_id:
                    local_path = paper_data_item.get("local_path", "")
                    paper_id = Path(local_path).name if local_path else f"item_index_{idx}"

                if paper_id in processed_ids:
                    continue

                session_start_time = time.time()

                try:
                    # This call now returns two items
                    paper_specific_material_entries, current_paper_stats = \
                        self.process_single_paper_llm_call(paper_data_item, domain_name, language)

                    all_paper_stats.append(current_paper_stats)  # Collect stats for this paper

                    if paper_specific_material_entries:
                        all_extracted_results.extend(paper_specific_material_entries)
                        logger.info(
                            f"Successfully processed Paper ID {paper_id}, found {len(paper_specific_material_entries)} valid material entries.")
                    # Even if no valid entries, paper was "processed" in terms of an attempt

                    processed_ids.add(paper_id)
                    newly_processed_paper_count_this_session += 1
                    # ... (ETA calculation and print logic as before) ...
                    current_total_processed_papers = num_already_processed_papers + newly_processed_paper_count_this_session
                    session_elapsed_time = time.time() - session_start_time
                    total_elapsed_time += session_elapsed_time

                    avg_time_per_paper = total_elapsed_time / current_total_processed_papers if current_total_processed_papers > 0 else 0
                    remaining_papers = total_papers_to_process - current_total_processed_papers
                    eta_seconds = avg_time_per_paper * remaining_papers if remaining_papers > 0 else 0

                    print(
                        f"\r✔ Extr. Progress [{current_total_processed_papers}/{total_papers_to_process}] | "
                        f"Paper ID: {str(paper_id)[:30]}... | "
                        f"Last: {session_elapsed_time:.1f}s | "
                        f"Total Time: {total_elapsed_time / 3600:.2f}h | "
                        f"ETA: {eta_seconds / 3600:.1f}h",
                        end="", flush=True
                    )

                    if newly_processed_paper_count_this_session > 0 and newly_processed_paper_count_this_session % 5 == 0:
                        checkpoint_to_save = {
                            "processed_ids": processed_ids,  # This is a set
                            "results": all_extracted_results,
                            "all_paper_stats": all_paper_stats,  # Save collected stats
                            "total_elapsed": total_elapsed_time,
                            "total_processed": current_total_processed_papers
                        }
                        save_checkpoint(checkpoint_file_path,
                                        checkpoint_to_save)  # file_operations.py handles set->list for processed_ids
                        logger.debug("Intermediate extraction checkpoint saved.")

                except KeyboardInterrupt:  # Inner try
                    logger.warning("\nKeyboard interrupt during paper processing. Saving progress...")
                    raise
                except Exception as e:  # Inner try
                    logger.error(f"\nError processing Paper ID {paper_id}: {e}", exc_info=True)
                    # Log basic failure stat if an unexpected error occurs before parse_llm_response finishes
                    all_paper_stats.append(
                        {"paper_id": paper_id, "top_level_error_message": f"Core processor error: {str(e)}"})
                    continue

        except KeyboardInterrupt:  # Outer try
            pass  # Progress should be saved by finally
        except Exception as e:  # Outer try
            logger.critical(f"\nUnhandled exception in batch processing: {e}", exc_info=True)
        finally:
            final_checkpoint_data = {
                "processed_ids": processed_ids,  # This is a set
                "results": all_extracted_results,
                "all_paper_stats": all_paper_stats,  # Save collected stats
                "total_elapsed": total_elapsed_time,
                "total_processed": num_already_processed_papers + newly_processed_paper_count_this_session
            }
            save_checkpoint(checkpoint_file_path,
                            final_checkpoint_data)  # file_operations.py handles set->list for processed_ids
            logger.info("Final extraction checkpoint saved.")

        # ... (cleanup_checkpoint logic as before) ...
        fully_processed_count = num_already_processed_papers + newly_processed_paper_count_this_session
        if total_papers_to_process > 0 and fully_processed_count == total_papers_to_process:
            cleanup_checkpoint(checkpoint_file_path)
        elif total_papers_to_process == 0:
            logger.info("No papers were scheduled for processing. Checkpoint not cleaned for extraction stats.")

        # ... (logging for completion) ...
        return all_extracted_results, all_paper_stats  # Return both results and stats