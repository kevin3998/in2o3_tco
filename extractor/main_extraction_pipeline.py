"""
Module: main_extraction_pipeline
Functionality: Serves as the main entry point for running the information
               extraction pipeline. It handles command-line argument parsing,
               initializes necessary components (LLM client, configurations,
               processor), loads input data, orchestrates the processing,
               and saves the final results.
"""
import argparse
import logging
from extractor.utils.logging_config import setup_logging
from extractor.utils.llm_client_setup import get_openai_client
from extractor.utils.file_operations import load_json_data, save_json_data
from extractor.utils.general_utils import PromptManager
from extractor.config.domain_specific_configs import get_domain_config
from extractor.config.prompt_templates import load_prompts as load_all_extraction_prompts # Renamed for clarity
from extractor.extraction.core_processor import PaperProcessor

# Setup logging as early as possible
setup_logging(level=logging.INFO) # Set to logging.DEBUG for more verbosity
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the academic paper information extraction pipeline.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing paper data.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the extracted structured information.")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint_extraction.json", help="Path for the checkpoint file.")
    parser.add_argument("--domain", type=str, default="membrane", help="The domain for extraction (e.g., 'membrane').")
    parser.add_argument("--language", type=str, default="en", help="Language of the papers and prompts (e.g., 'en').")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-671B", help="Name of the LLM to use for extraction.") # Or your preferred default

    args = parser.parse_args()

    logger.info(f"Starting extraction pipeline with a_args: {args}")

    try:
        # 1. Initialize LLM Client
        openai_client = get_openai_client()

        # 2. Load Domain Configuration
        domain_config = get_domain_config(args.domain)
        logger.info(f"Loaded configuration for domain: {args.domain}")

        # 3. Initialize PromptManager and load prompts
        prompt_manager = PromptManager()
        load_all_extraction_prompts(prompt_manager) # This function should populate the manager
        logger.info(f"Prompts loaded into PromptManager. Available for: {prompt_manager.templates.keys()}")


        # 4. Initialize PaperProcessor
        processor = PaperProcessor(
            client=openai_client,
            prompt_manager=prompt_manager,
            domain_config=domain_config,
            model_name=args.model_name
        )
        logger.info(f"PaperProcessor initialized with model: {args.model_name}")

        # 5. Load input paper data
        papers_data_list = load_json_data(args.input_file)
        if not isinstance(papers_data_list, list):
            logger.error(f"Input file {args.input_file} does not contain a list of papers.")
            return
        logger.info(f"Loaded {len(papers_data_list)} papers from {args.input_file}")

        # 6. Process papers
        extracted_results = processor.process_papers_with_checkpoint(
            papers_list=papers_data_list,
            domain_name=args.domain,
            language=args.language,
            checkpoint_file_path=args.checkpoint_file
        )

        # 7. Save results
        save_json_data(extracted_results, args.output_file)
        logger.info(f"Extraction complete. Results saved to {args.output_file}")

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
    except FileNotFoundError as fnfe:
        logger.error(f"File access error: {fnfe}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()