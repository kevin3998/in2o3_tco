# src/extraction/response_parser.py
import regex as re
import json
from typing import List, Dict, Any
from extractor.extraction.field_standardizer import (
    standardize_field_names_in_details,
    ensure_required_sections,
    clean_material_name,
    recursive_standardize_keys,
    extract_material_from_entry_dict
)
from extractor.config.domain_specific_configs import DomainConfig
from extractor.extraction.schemas import (
    LLMOutputSchema, # For overall LLM output structure
    # Domain-specific Detail Schemas - Import TCO and potentially Membrane if you have it
    TCOSpecificDetailsSchema,
    # MembraneSpecificDetailsSchema, # Assuming you'd have a similar one for membranes
    # ExtractedMaterialEntrySchema # This is used by LLMOutputSchema, not directly called here for top-level validation
)
from pydantic import ValidationError # Import ValidationError directly from pydantic
import logging

logger = logging.getLogger(__name__)

def parse_llm_response(
    llm_content: str,
    original_input_text_for_llm: str,
    paper_meta: Dict,
    domain_config: DomainConfig,
    language: str
) -> List[Dict]:
    parsed_material_entries_for_paper = [] # Stores final structured entries for THIS paper
    
    meta_doi = paper_meta.get("doi", "N/A_DOI")
    meta_title = paper_meta.get("retrieved_title", "Unknown Title")
    meta_journal = paper_meta.get("retrieved_journal", "Unknown Journal")
    meta_year = paper_meta.get("retrieved_year", "Unknown Year")

    try:
        json_str_cleaned = re.sub(r"[\x00-\x1F]", "", llm_content)
        match = re.search(r"\{(?:[^{}]|(?R))*\}", json_str_cleaned, re.DOTALL)
        
        if not match:
            logger.warning(f"No valid JSON object found in LLM response for DOI {meta_doi}. Content: {llm_content[:300]}")
            return []
        
        json_block_str = match.group(0)
        
        try:
            raw_data_from_llm = json.loads(json_block_str, strict=False)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError while parsing main block for DOI {meta_doi}: {e}. Block: {json_block_str[:300]}")
            return []

        # Standardize keys of the entire parsed object (e.g., "output" -> "Output")
        standardized_llm_output_obj = recursive_standardize_keys(raw_data_from_llm)

        # --- 1. Validate Overall LLM Output Structure ---
        try:
            # LLMOutputSchema expects a top-level dictionary with an "Output" key,
            # where "Output" is a list of ExtractedMaterialEntrySchema-like dictionaries.
            validated_llm_root = LLMOutputSchema.model_validate(standardized_llm_output_obj)
        except ValidationError as e_root:
            logger.error(f"Pydantic Validation Error for overall LLM output structure (DOI {meta_doi}): {e_root.errors()}")
            logger.debug(f"Data that failed LLMOutputSchema validation: {standardized_llm_output_obj}")
            return []
        except ImportError: # Should not happen if schemas.py exists
            logger.critical("Pydantic schemas.py not found or LLMOutputSchema missing. Cannot validate.")
            return []

        # --- 2. Process and Validate Each Material Entry ---
        # validated_llm_root.Output is a list of ExtractedMaterialEntrySchema Pydantic objects
        for entry_model in validated_llm_root.Output:
            # entry_model.MaterialName is already validated by ExtractedMaterialEntrySchema
            # entry_model.Details is a Dict[str, Any] at this stage, as defined in ExtractedMaterialEntrySchema
            
            details_dict_from_llm = entry_model.Details # This is a plain Python dict

            # Apply your existing standardization to the keys WITHIN this details_dict_from_llm
            # This ensures keys like "design" become "Design", ready for domain-specific Pydantic validation.
            # The Pydantic aliases (e.g., alias="HostMaterial") expect these standardized keys.
            standardized_sub_details = standardize_field_names_in_details(details_dict_from_llm, language, domain_config)
            final_details_for_domain_validation = ensure_required_sections(standardized_sub_details)
            
            validated_domain_specific_details_model = None
            details_to_store_in_final_output = final_details_for_domain_validation # Default

            try:
                current_domain = domain_config.domain

                if current_domain == "in2o3_tco":
                    validated_domain_specific_details_model = TCOSpecificDetailsSchema.model_validate(final_details_for_domain_validation)
                else:
                    logger.warning(f"No specific Pydantic 'Details' schema defined for domain: {current_domain}. Storing standardized dictionary for DOI {meta_doi}.")

                # If domain-specific validation was performed and successful, convert model to dict
                if validated_domain_specific_details_model:
                    details_to_store_in_final_output = validated_domain_specific_details_model.model_dump(
                        exclude_none=True, # Don't include fields that are None
                        by_alias=True      # IMPORTANT: Use JSON key aliases (e.g., "HostMaterial")
                                           # instead of Python attribute names (e.g., "host_material")
                    )

            except ValidationError as e_details:
                logger.error(f"Pydantic Validation Error for '{current_domain}' Details (Material: {entry_model.MaterialName}, DOI {meta_doi}): {e_details.errors()}")
                logger.debug(f"Details data that failed domain-specific validation for '{current_domain}': {final_details_for_domain_validation}")
                continue # Skip this material entry if its specific details are invalid

            # Construct the final structured object for THIS material entry
            parsed_material_entries_for_paper.append({
                "meta_source_paper": {
                    "doi": meta_doi,
                    "title": meta_title,
                    "journal": meta_journal,
                    "year": meta_year,
                    "original_filename": paper_meta.get("filename", ""),
                    "local_path": paper_meta.get("local_path", "")
                },
                "llm_input_text_segment": original_input_text_for_llm,
                "extracted_material_data": {
                    "MaterialName": entry_model.MaterialName, # From validated ExtractedMaterialEntrySchema
                    "Details": details_to_store_in_final_output # Validated and dumped domain-specific details
                }
            })
            
    except Exception as e: # Catch any other unexpected errors during parsing
        logger.error(f"Overall parsing or validation failure for DOI {meta_doi}: {e}", exc_info=True)
        return []

    return parsed_material_entries_for_paper