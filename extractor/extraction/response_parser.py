# extractor/extraction/response_parser.py
import regex as re
import json
from typing import List, Dict, Any, Tuple  # Added Tuple
from .field_standardizer import (
    standardize_field_names_in_details,
    ensure_required_sections,
    clean_material_name,
    recursive_standardize_keys,
    extract_material_from_entry_dict
)
from ..config.domain_specific_configs import DomainConfig
from .schemas import (
    LLMOutputSchema,
    TCOSpecificDetailsSchema,  # Assuming this is the primary one for now
    # MembraneSpecificDetailsSchema, # Add if you have this defined
)
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


def parse_llm_response(
        llm_content: str,
        original_input_text_for_llm: str,
        paper_meta: Dict,  # This is an item from your PDF processor's output list
        domain_config: DomainConfig,
        language: str
) -> Tuple[List[Dict], Dict[str, Any]]:  # Return type changed
    """
    Parses LLM response, performs Pydantic validation, and collects statistics.
    Returns:
        Tuple[List[Dict], Dict[str, Any]]:
            - List of successfully extracted and validated material entries.
            - Dictionary containing validation statistics for this paper.
    """
    parsed_material_entries_for_paper = []
    paper_id = paper_meta.get("doi", paper_meta.get("filename", "unknown_paper"))

    # Initialize stats for this paper
    paper_stats = {
        "paper_id": paper_id,
        "raw_llm_response_parseable_as_json": False,
        "initial_entry_count_from_llm": 0,
        "top_level_validation_passed": False,
        "top_level_error_message": None,
        "num_entries_after_top_level_validation": 0,  # Entries structurally conforming to ExtractedMaterialEntrySchema
        "domain_specific_validation_results": [],  # List of (material_name, passed_bool, error_str)
        "count_entries_passing_domain_details_validation": 0
    }

    meta_doi = paper_meta.get("doi", "N/A_DOI")  # Used for logging within this function
    # ... (other meta like meta_title, meta_journal, meta_year for constructing output)

    try:
        json_str_cleaned = re.sub(r"[\x00-\x1F]", "", llm_content)
        match = re.search(r"\{(?:[^{}]|(?R))*\}", json_str_cleaned, re.DOTALL)

        if not match:
            logger.warning(
                f"No valid JSON object found in LLM response for Paper ID {paper_id}. Content: {llm_content[:300]}")
            return [], paper_stats  # Return empty list and current stats

        json_block_str = match.group(0)

        try:
            raw_data_from_llm = json.loads(json_block_str, strict=False)
            paper_stats["raw_llm_response_parseable_as_json"] = True
        except json.JSONDecodeError as e:
            logger.error(
                f"JSONDecodeError while pars ing main block for Paper ID {paper_id}: {e}. Block: {json_block_str[:300]}")
            paper_stats["top_level_error_message"] = f"JSONDecodeError: {str(e)}"
            return [], paper_stats

        standardized_llm_output_obj = recursive_standardize_keys(raw_data_from_llm)
        paper_stats["initial_entry_count_from_llm"] = len(standardized_llm_output_obj.get("Output", []))

        # --- 1. Top-level Schema Validation (LLMOutputSchema) ---
        validated_llm_root = None
        try:
            validated_llm_root = LLMOutputSchema.model_validate(standardized_llm_output_obj)
            paper_stats["top_level_validation_passed"] = True
            paper_stats["num_entries_after_top_level_validation"] = len(validated_llm_root.Output)
        except ValidationError as e_root:
            logger.error(
                f"Pydantic Validation Error for overall LLM output structure (Paper ID {paper_id}): {e_root.errors()}")
            paper_stats["top_level_error_message"] = str(e_root.errors())
            # No further processing if top-level fails
            return [], paper_stats
        except ImportError:
            logger.critical("Pydantic schemas.py not found or LLMOutputSchema missing. Cannot validate.")
            paper_stats["top_level_error_message"] = "ImportError: Pydantic schemas missing."
            return [], paper_stats

        # --- 2. Process and Validate Each Material Entry's Details ---
        for entry_model in validated_llm_root.Output:  # entry_model is ExtractedMaterialEntrySchema instance
            material_name_for_stats = entry_model.MaterialName  # For logging/stats
            entry_domain_details_passed = False
            entry_domain_details_error = None

            details_dict_from_llm = entry_model.Details

            standardized_sub_details = standardize_field_names_in_details(details_dict_from_llm, language,
                                                                          domain_config)
            final_details_for_domain_validation = ensure_required_sections(standardized_sub_details)

            validated_domain_specific_details_model = None
            details_to_store_in_final_output = final_details_for_domain_validation

            try:
                current_domain = domain_config.domain
                if current_domain == "membrane":
                    # Assuming MembraneSpecificDetailsSchema is imported and defined
                    # from .schemas import MembraneSpecificDetailsSchema
                    # validated_domain_specific_details_model = MembraneSpecificDetailsSchema.model_validate(final_details_for_domain_validation)
                    logger.warning(
                        f"MembraneSpecificDetailsSchema actual validation not fully implemented in this stats collection example for Paper ID {paper_id}.")
                    entry_domain_details_passed = True  # Placeholder - assume pass if not implemented
                elif current_domain == "in2o3_tco":
                    validated_domain_specific_details_model = TCOSpecificDetailsSchema.model_validate(
                        final_details_for_domain_validation)
                    entry_domain_details_passed = True
                else:
                    logger.warning(
                        f"No specific Pydantic 'Details' schema defined for domain: {current_domain}. Storing standardized dictionary for Paper ID {paper_id}, Material: {material_name_for_stats}.")
                    entry_domain_details_passed = True  # Considered "passing" as no specific schema to fail against

                if validated_domain_specific_details_model:
                    details_to_store_in_final_output = validated_domain_specific_details_model.model_dump(
                        exclude_none=True, by_alias=True
                    )

            except ValidationError as e_details:
                logger.error(
                    f"Pydantic Validation Error for '{current_domain}' Details (Material: {material_name_for_stats}, Paper ID {paper_id}): {e_details.errors()}")
                entry_domain_details_error = str(e_details.errors())
                # continue # Don't skip here, just record failure for stats; skipping happens if not added to parsed_material_entries_for_paper

            paper_stats["domain_specific_validation_results"].append({
                "material_name": material_name_for_stats,
                "passed": entry_domain_details_passed,
                "error_message": entry_domain_details_error
            })

            if entry_domain_details_passed:
                paper_stats["count_entries_passing_domain_details_validation"] += 1
                # Construct the final output for this material entry
                parsed_material_entries_for_paper.append({
                    "meta_source_paper": {
                        "doi": paper_meta.get("doi", "N/A_DOI"),  # Use meta_doi if preferred
                        "title": paper_meta.get("retrieved_title", "Unknown Title"),
                        "journal": paper_meta.get("retrieved_journal", "Unknown Journal"),
                        "year": paper_meta.get("retrieved_year", "Unknown Year"),
                        "original_filename": paper_meta.get("filename", ""),
                        "local_path": paper_meta.get("local_path", "")
                    },
                    "llm_input_text_segment": original_input_text_for_llm,
                    "extracted_material_data": {
                        "MaterialName": entry_model.MaterialName,
                        "Details": details_to_store_in_final_output
                    }
                })
            # If entry_domain_details_passed is False, this entry is not added to parsed_material_entries_for_paper

    except Exception as e:
        logger.error(f"Overall parsing or Pydantic validation failure for Paper ID {paper_id}: {e}", exc_info=True)
        # Ensure top_level_error_message is set if it wasn't a Pydantic error initially
        if paper_stats["raw_llm_response_parseable_as_json"] and not paper_stats["top_level_error_message"]:
            paper_stats["top_level_error_message"] = f"General Exception: {str(e)}"
        return [], paper_stats

    return parsed_material_entries_for_paper, paper_stats