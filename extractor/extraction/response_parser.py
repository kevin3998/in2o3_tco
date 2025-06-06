# src/extraction/response_parser.py
import regex as re
import json
import time
from typing import List, Dict, Any, Tuple
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
    TCOSpecificDetailsSchema,
    # MembraneSpecificDetailsSchema, # Ensure this is defined if used
)
from pydantic import ValidationError, BaseModel
import logging

logger = logging.getLogger(__name__)


def parse_llm_response(
        llm_content: str,
        original_input_text_for_llm: str,
        paper_meta: Dict,
        domain_config: DomainConfig,
        language: str,
        pydantic_validation_enabled: bool = True
) -> Tuple[List[Dict], Dict[str, Any]]:
    parsed_material_entries_for_paper = []
    paper_id = paper_meta.get("doi", paper_meta.get("filename", "unknown_paper"))

    paper_stats = {
        "paper_id": paper_id,
        "raw_llm_response_parseable_as_json": False,
        "json_decode_error": None,
        "initial_entry_count_from_llm": 0,
        "top_level_validation_passed": False if pydantic_validation_enabled else "SKIPPED",
        "top_level_validation_errors": None,
        "time_top_level_validation_seconds": 0.0,
        "num_entries_after_top_level_validation": 0,
        "domain_specific_validation_results": [],
        "count_entries_passing_domain_details_validation": 0,
        "overall_parsing_exception": None
    }

    meta_doi = paper_meta.get("doi", "N/A_DOI")
    # meta_title, etc. for constructing final output
    meta_title = paper_meta.get("retrieved_title", "Unknown Title")
    meta_journal = paper_meta.get("retrieved_journal", "Unknown Journal")
    meta_year = paper_meta.get("retrieved_year", "Unknown Year")

    try:
        json_str_cleaned = re.sub(r"[\x00-\x1F]", "", llm_content)
        match = re.search(r"\{(?:[^{}]|(?R))*\}", json_str_cleaned, re.DOTALL)

        if not match:
            logger.warning(
                f"No valid JSON object found in LLM response for Paper ID {paper_id}. Content: {llm_content[:300]}")
            paper_stats["json_decode_error"] = "No valid JSON object found in LLM response string."
            return [], paper_stats

        json_block_str = match.group(0)

        try:
            raw_data_from_llm = json.loads(json_block_str, strict=False)
            paper_stats["raw_llm_response_parseable_as_json"] = True
        except json.JSONDecodeError as e:
            logger.error(
                f"JSONDecodeError while parsing main block for Paper ID {paper_id}: {e}. Block: {json_block_str[:300]}")
            paper_stats["json_decode_error"] = f"JSONDecodeError: {str(e)}"
            return [], paper_stats

        standardized_llm_output_obj = recursive_standardize_keys(raw_data_from_llm)
        paper_stats["initial_entry_count_from_llm"] = len(standardized_llm_output_obj.get("Output", []))

        material_entries_to_iterate = []

        if pydantic_validation_enabled:
            time_start_top_val = time.time()
            try:
                validated_llm_root = LLMOutputSchema.model_validate(standardized_llm_output_obj)
                paper_stats["top_level_validation_passed"] = True
                paper_stats["num_entries_after_top_level_validation"] = len(validated_llm_root.Output)
                material_entries_to_iterate = validated_llm_root.Output
            except ValidationError as e_root:
                logger.error(
                    f"Pydantic Validation Error for overall LLM output structure (Paper ID {paper_id}): {e_root.errors()}")
                paper_stats["top_level_validation_errors"] = e_root.errors()
                return [], paper_stats
            finally:
                paper_stats["time_top_level_validation_seconds"] = time.time() - time_start_top_val
        else:
            logger.info(f"Pydantic Top-Level Validation SKIPPED for Paper ID {paper_id}")
            material_entries_to_iterate = standardized_llm_output_obj.get("Output", [])
            if not isinstance(material_entries_to_iterate, list):
                material_entries_to_iterate = []
            paper_stats["num_entries_after_top_level_validation"] = len(material_entries_to_iterate)

        for entry_data in material_entries_to_iterate:
            material_name_for_stats = ""
            details_dict_for_standardization = {}

            if pydantic_validation_enabled and isinstance(entry_data, BaseModel):
                material_name_for_stats = entry_data.MaterialName
                details_dict_for_standardization = entry_data.Details
            elif isinstance(entry_data, dict):
                material_name_for_stats = clean_material_name(extract_material_from_entry_dict(entry_data))
                details_dict_for_standardization = entry_data.get("Details", {})
            else:
                logger.warning(f"Skipping invalid entry data type: {type(entry_data)} for Paper ID {paper_id}")
                continue

            if not isinstance(details_dict_for_standardization, dict):
                details_dict_for_standardization = {}
                logger.warning(
                    f"Details for material {material_name_for_stats} (Paper ID {paper_id}) was not a dict. Using empty.")

            entry_domain_details_passed_status = False if pydantic_validation_enabled else "SKIPPED"
            entry_domain_details_error_list = None
            time_details_validation_seconds = 0.0

            standardized_sub_details = standardize_field_names_in_details(details_dict_for_standardization, language,
                                                                          domain_config)
            final_details_for_processing = ensure_required_sections(standardized_sub_details)

            details_to_store_in_final_output = final_details_for_processing

            if pydantic_validation_enabled:
                time_start_details_val = time.time()
                try:
                    current_domain = domain_config.domain
                    validated_domain_specific_details_model = None
                    if current_domain == "membrane":
                        logger.warning(
                            f"MembraneSpecificDetailsSchema validation not fully implemented. Paper ID {paper_id}.")
                        entry_domain_details_passed_status = True
                    elif current_domain == "in2o3_tco":
                        validated_domain_specific_details_model = TCOSpecificDetailsSchema.model_validate(
                            final_details_for_processing)
                        entry_domain_details_passed_status = True
                    else:
                        logger.warning(
                            f"No specific Pydantic 'Details' schema for domain: {current_domain}. Paper ID {paper_id}, Material: {material_name_for_stats}.")
                        entry_domain_details_passed_status = True

                    if validated_domain_specific_details_model:
                        details_to_store_in_final_output = validated_domain_specific_details_model.model_dump(
                            exclude_none=True, by_alias=True)
                except ValidationError as e_details:
                    logger.error(
                        f"Pydantic Validation Error for '{current_domain}' Details (Material: {material_name_for_stats}, Paper ID {paper_id}): {e_details.errors()}")
                    entry_domain_details_error_list = e_details.errors()
                    entry_domain_details_passed_status = False
                finally:
                    time_details_validation_seconds = time.time() - time_start_details_val

            paper_stats["domain_specific_validation_results"].append({
                "material_name": material_name_for_stats,
                "passed": entry_domain_details_passed_status,
                "validation_errors": entry_domain_details_error_list,
                "time_details_validation_seconds": time_details_validation_seconds
            })

            if entry_domain_details_passed_status is True or entry_domain_details_passed_status == "SKIPPED":
                paper_stats["count_entries_passing_domain_details_validation"] += 1
                parsed_material_entries_for_paper.append({
                    "meta_source_paper": {
                        "doi": paper_meta.get("doi", "N/A_DOI"),
                        "title": meta_title,
                        "journal": meta_journal,
                        "year": meta_year,
                        "original_filename": paper_meta.get("filename", ""),
                        "local_path": paper_meta.get("local_path", "")
                    },
                    "llm_input_text_segment": original_input_text_for_llm,
                    "extracted_material_data": {
                        "MaterialName": material_name_for_stats,
                        "Details": details_to_store_in_final_output
                    }
                })

    except Exception as e:
        logger.error(f"Overall parsing or Pydantic validation failure for Paper ID {paper_id}: {e}", exc_info=True)
        paper_stats["overall_parsing_exception"] = str(e)
        if paper_stats["raw_llm_response_parseable_as_json"] and paper_stats[
            "top_level_validation_passed"] is True and not paper_stats["top_level_validation_errors"]:
            paper_stats["top_level_validation_errors"] = [{"type": type(e).__name__, "msg": str(e)}]
        return [], paper_stats

    return parsed_material_entries_for_paper, paper_stats