# src/extraction/core_processor.py
import time
from typing import List, Dict, Any
from zipfile import Path

from openai import OpenAI
from extractor.utils.general_utils import PromptManager
from extractor.config.domain_specific_configs import DomainConfig
from extractor.extraction.response_parser import parse_llm_response
from extractor.utils.file_operations import load_checkpoint, save_checkpoint, cleanup_checkpoint
import logging
import os

logger = logging.getLogger(__name__)

class PaperProcessor:
    def __init__(self, client: OpenAI, prompt_manager: PromptManager, domain_config: DomainConfig, model_name: str = "gpt-4"):
        self.client = client
        self.prompt_manager = prompt_manager
        self.domain_config = domain_config
        self.model_name = model_name

    def _get_text_for_llm(self, paper_data: Dict) -> str:
        """
        Retrieves the primary text content for LLM processing from the structured paper data.
        The input `paper_data` is an item from the output of your new PDF processing script.
        """
        # Primary source of text is "llm_ready_fulltext_cleaned"
        llm_text = paper_data.get("llm_ready_fulltext_cleaned", "")
        if isinstance(llm_text, str) and llm_text.strip():
            return llm_text.strip()

        # Fallback to abstract if llm_ready_fulltext_cleaned is missing or empty
        abstract_text = paper_data.get("extracted_abstract_cleaned", "")
        if isinstance(abstract_text, str) and abstract_text.strip():
            logger.info(
                f"Using 'extracted_abstract_cleaned' for paper DOI: {paper_data.get('doi', 'N/A')} "
                f"as 'llm_ready_fulltext_cleaned' is missing or empty."
            )
            return abstract_text.strip()

        logger.warning(
            f"No usable text found ('llm_ready_fulltext_cleaned' or 'extracted_abstract_cleaned') "
            f"for paper DOI: {paper_data.get('doi', 'N/A')}, "
            f"Title: {paper_data.get('retrieved_title', 'N/A')}. "
            f"Available keys in paper_data: {list(paper_data.keys())}. Returning empty string."
        )
        return ""

    def process_single_paper_llm_call(self, paper_data: Dict, domain_name: str, language: str) -> List[Dict]:
        # `paper_data` is now one item from the output of your PDF processing script
        text_to_process = self._get_text_for_llm(paper_data)
        if not text_to_process:
            # Logging for this case is already handled in _get_text_for_llm
            logger.warning(f"Skipping paper DOI {paper_data.get('doi', 'N/A')} (Title: {paper_data.get('retrieved_title', 'N/A')}) due to empty text for LLM.")
            return []

        # The prompt will be formatted with text_to_process
        prompt_for_llm = self.prompt_manager.get_prompt(domain_name, language, text_to_process)

        max_retries = 3
        # extracted_results will be a list of material entries for THIS paper
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} for paper DOI: {paper_data.get('doi', 'N/A')}")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert academic extraction assistant specialized in materials science. Adhere strictly to the user's requested JSON output format."},
                        {"role": "user", "content": prompt_for_llm}
                    ],
                    temperature=0.1,
                    stream=False
                )
                llm_response_content = response.choices[0].message.content

                # Pass the full paper_data item as paper_meta, as it contains all metadata
                # Also pass text_to_process as original_input_text for the parser
                extracted_material_entries = parse_llm_response(
                    llm_content=llm_response_content,
                    original_input_text_for_llm=text_to_process, # This is the text LLM actually saw
                    paper_meta=paper_data, # Pass the whole processed PDF data object
                    domain_config=self.domain_config,
                    language=language
                )
                return extracted_material_entries

            except Exception as e:
                logger.error(f"LLM call or parsing failed for DOI {paper_data.get('doi', 'N/A')} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Final attempt failed for DOI {paper_data.get('doi', 'N/A')}. Skipping this paper.")
                    return []
        return []

    def process_papers_with_checkpoint(self,
                                       papers_list: List[Dict], # This list now contains objects from your PDF processor
                                       domain_name: str,
                                       language: str,
                                       checkpoint_file_path: str = "checkpoint_extraction.json"
                                       ) -> List[Dict]:
        default_checkpoint_data = {
            "processed_ids": set(), # Using a generic "ids" now, will store DOI or filename
            "results": [],
            "total_elapsed": 0.0,
            "total_processed": 0
        }
        checkpoint_data = load_checkpoint(checkpoint_file_path, default_checkpoint_data)

        # Adjust key name for checkpoint if needed, e.g. "processed_dois" to "processed_ids"
        # Assuming load_checkpoint handles 'processed_dois' vs 'processed_ids' gracefully or is updated
        processed_ids = checkpoint_data.get("processed_ids", checkpoint_data.get("processed_dois", set()))
        all_extracted_results = checkpoint_data["results"] # This will be a list of material_entries
        total_elapsed_time = checkpoint_data["total_elapsed"]
        num_already_processed_papers = checkpoint_data["total_processed"]

        total_papers_to_process = len(papers_list)

        logger.info(f"Starting batch processing for structured extraction. Total input items: {total_papers_to_process}. Already processed from checkpoint: {num_already_processed_papers}.")

        newly_processed_paper_count_this_session = 0

        try:
            for idx, paper_data_item in enumerate(papers_list):
                # Use DOI if available, otherwise filename from local_path as a unique ID for the paper_data_item
                paper_id = paper_data_item.get("doi", None)
                if not paper_id: # If DOI is None or empty string
                    local_path = paper_data_item.get("local_path", "")
                    paper_id = Path(local_path).name if local_path else f"item_index_{idx}"

                if paper_id in processed_ids:
                    continue

                session_start_time = time.time()

                try:
                    # This call returns a list of structured data for EACH material found in this single paper
                    paper_specific_material_entries = self.process_single_paper_llm_call(paper_data_item, domain_name, language)

                    if paper_specific_material_entries:
                        all_extracted_results.extend(paper_specific_material_entries) # Add all found material entries
                        logger.info(f"Successfully processed Paper ID {paper_id}, found {len(paper_specific_material_entries)} material entries.")
                    else:
                        logger.warning(f"No results extracted for Paper ID {paper_id} after LLM call and parsing.")

                    processed_ids.add(paper_id)
                    newly_processed_paper_count_this_session += 1
                    current_total_processed_papers = num_already_processed_papers + newly_processed_paper_count_this_session

                    session_elapsed_time = time.time() - session_start_time
                    total_elapsed_time += session_elapsed_time

                    avg_time_per_paper = total_elapsed_time / current_total_processed_papers if current_total_processed_papers > 0 else 0
                    remaining_papers = total_papers_to_process - current_total_processed_papers
                    eta_seconds = avg_time_per_paper * remaining_papers if remaining_papers > 0 else 0

                    print(
                        f"\r✔ Extr. Progress [{current_total_processed_papers}/{total_papers_to_process}] | "
                        f"Paper ID: {str(paper_id)[:30]}... | " # Ensure paper_id is str for slicing
                        f"Last: {session_elapsed_time:.1f}s | "
                        f"Total Time: {total_elapsed_time/3600:.2f}h | "
                        f"ETA: {eta_seconds / 3600:.1f}h",
                        end="", flush=True
                    )

                    if newly_processed_paper_count_this_session > 0 and newly_processed_paper_count_this_session % 5 == 0:
                        checkpoint_to_save = {
                            "processed_ids": processed_ids, # Save the set of processed paper IDs
                            "results": all_extracted_results,
                            "total_elapsed": total_elapsed_time,
                            "total_processed": current_total_processed_papers
                        }
                        save_checkpoint(checkpoint_file_path, checkpoint_to_save)
                        logger.debug("Intermediate extraction checkpoint saved.")

                except KeyboardInterrupt:
                    logger.warning("\nKeyboard interrupt detected during paper processing. Saving progress...")
                    raise
                except Exception as e:
                    logger.error(f"\nError processing Paper ID {paper_id}: {e}", exc_info=True)
                    continue

        except KeyboardInterrupt:
             pass
        except Exception as e:
            logger.critical(f"\nUnhandled exception in batch processing: {e}", exc_info=True)
        finally:
            final_checkpoint_data = {
                "processed_ids": processed_ids,
                "results": all_extracted_results,
                "total_elapsed": total_elapsed_time,
                "total_processed": num_already_processed_papers + newly_processed_paper_count_this_session
            }
            save_checkpoint(checkpoint_file_path, final_checkpoint_data)
            logger.info("Final extraction checkpoint saved.")

        print()
        fully_processed_count = num_already_processed_papers + newly_processed_paper_count_this_session
        if total_papers_to_process > 0 and fully_processed_count == total_papers_to_process:
            cleanup_checkpoint(checkpoint_file_path)
        elif total_papers_to_process == 0:
             logger.info("No papers were scheduled for processing. Checkpoint not cleaned.")

        logger.info(f"Extraction batch processing completed. Total unique papers processed in this run: {newly_processed_paper_count_this_session}")
        logger.info(f"Total extracted material entries in results: {len(all_extracted_results)}.") # This count is material entries
        logger.info(f"Total cumulative time for extraction: {total_elapsed_time / 3600:.2f} hours.")
        return all_extracted_results