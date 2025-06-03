"""
Module: field_standardizer
Functionality: Provides functions for standardizing field names and values
               extracted by the LLM. This includes cleaning material names,
               mapping LLM output field names to canonical internal names,
               ensuring required sections are present in the output, and
               recursively standardizing dictionary keys.
"""
import regex as re
from typing import List, Dict, Any
from extractor.config.domain_specific_configs import DomainConfig  # Corrected import path
import logging

logger = logging.getLogger(__name__)


def standardize_field_names_in_details(details_dict: Dict, lang: str, domain_config: DomainConfig) -> Dict:
    """
    Standardizes keys within the 'Details' sub-dictionaries (Design, Fabrication, etc.)
    and the main keys of the Details object itself (e.g., "design" -> "Design").
    """
    if not isinstance(details_dict, dict):
        logger.warning(f"Details_dict is not a dictionary: {type(details_dict)}. Skipping standardization.")
        return details_dict

    standardized_details = {}
    # Standardize top-level keys like "Design", "Fabrication", etc.
    # Example: if LLM returns "design" or "Material Design", map it to "Design"
    top_level_mapping = domain_config.field_mapping.get(lang, {})

    # Create a reverse mapping for variations to standard keys
    # e.g., {"design": "Design", "Material Design": "Design"}
    reverse_top_level_map = {}
    for standard_key, variations in top_level_mapping.items():
        if isinstance(variations, list):  # Ensure variations is a list
            for var in variations:
                reverse_top_level_map[var.lower()] = standard_key  # map lowercase variation
                reverse_top_level_map[var] = standard_key  # map original variation
        else:  # if field_mapping has direct key:value for some reason (not list of variations)
            reverse_top_level_map[variations.lower()] = standard_key
            reverse_top_level_map[variations] = standard_key

    for raw_key, raw_value in details_dict.items():
        standard_key = reverse_top_level_map.get(raw_key.lower(), raw_key)  # Try lowercase match first
        standard_key = reverse_top_level_map.get(raw_key, standard_key)  # Try original case match

        # Capitalize first letter if no specific mapping found, but this might be too aggressive
        # standard_key = standard_key[0].upper() + standard_key[1:] if standard_key else standard_key

        if isinstance(raw_value, dict):
            # Recursively standardize sub-dictionary keys if needed,
            # or apply specific sub-field mappings if defined in domain_config
            # For now, we just assign the value. If sub-fields also need mapping,
            # this function would need to be more complex or field_mapping more detailed.
            standardized_details[standard_key] = raw_value
        else:
            standardized_details[standard_key] = raw_value

    return standardized_details


def ensure_required_sections(details_dict: Dict) -> Dict:
    """Ensures the presence of standard top-level sections in Details."""
    required_sections = ["Design", "Fabrication", "Performance", "Application"]
    final_details = {}
    for section in required_sections:
        # Find the section key, potentially with different casing from LLM output
        found_key = None
        for k in details_dict.keys():
            if k.lower() == section.lower():
                found_key = k
                break
        final_details[section] = details_dict.get(found_key, {})  # Use found_key if exists, else empty dict
    return final_details


def clean_material_name(material_text: str) -> str:
    if not isinstance(material_text, str):
        logger.warning(f"Material text is not a string: {material_text}. Returning as is.")
        return str(material_text)  # Or handle as error

    # Allow Unicode subscripts/superscripts, Greek letters, and chemical formula relevant chars
    # This regex aims to keep alphanumeric, hyphens, parentheses, and specific scientific chars.
    # It removes most other special symbols that are unlikely to be part of a material name.
    cleaned = re.sub(
        r"[^\w\s\-()\[\]₀₁₂₃₄₅₆₇₈₉上下αβγδεζηθικλμνξπρσςτυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ\.,%@/']",
        "",
        material_text
    )
    # Remove content in parentheses if they seem to be explanations/abbreviations not part of the core name
    # Example: "Polyvinylidene Fluoride (PVDF)" -> "Polyvinylidene Fluoride"
    # This can be aggressive, use with caution or make it more specific
    # cleaned = re.sub(r"\s*\([^)]+\)\s*$", "", cleaned.strip()).strip() # Only if at the end

    # Take portion before common delimiters if they represent lists or further details
    cleaned = cleaned.split("(")[0].split(",")[0].split(";")[0].strip()

    # Remove generic numeric suffixes that might be artifacts (e.g., "PVDF1", "PES23")
    # This is also heuristic and might remove legitimate numbers if not careful.
    # cleaned = re.sub(r"\b(\D+)\d+$", r"\1", cleaned) # Only if ends with digits after non-digits

    return cleaned if cleaned else "Unknown"


def recursive_standardize_keys(data: Any) -> Any:
    """Standardizes dictionary keys to a consistent format (e.g., PascalCase or as defined)."""
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = format_key(k)  # Standardize current key
            new_dict[new_key] = recursive_standardize_keys(v)  # Recurse for value
        return new_dict
    elif isinstance(data, list):
        return [recursive_standardize_keys(item) for item in data]
    return data


def format_key(key: str) -> str:
    """Helper to format a single key (e.g., to PascalCase)."""
    if not isinstance(key, str) or not key:
        return key
    if key.lower().replace(" ", "").replace("_", "") == "materialname":  # Specific common variations
        return "MaterialName"  # Or just "Material" if that's your standard

    # Simple PascalCase example (can be made more robust)
    # Remove leading/trailing whitespace, replace spaces/underscores with a single space for splitting
    # s = key.strip().replace('_', ' ').replace('-', ' ')
    # parts = s.split()
    # if not parts: return key # return original if key becomes empty
    # return "".join(part[0].upper() + part[1:].lower() for part in parts) # Example: "water flux" -> "WaterFlux"

    # Current script's logic: Capitalize first letter, rest lower, unless special case
    # This might be too simple if keys are like "COD_concentration" and you want "CODConecentration" or "CodConcentration"
    # For now, let's stick to a simpler capitalization for general keys not covered by field_mapping
    # and rely on field_mapping for primary keys like "Design", "Performance".

    # If not a known special key, just ensure first letter is capitalized if it's a category like Design/Fabrication.
    # The field_mapping in standardize_field_names_in_details is more important for top-level detail keys.
    # This generic format_key is used in recursive_standardize_keys for all nested keys.
    if len(key) > 1:
        return key[0].upper() + key[1:]  # Example: "structure" -> "Structure", "key Parameters" -> "Key Parameters"
    elif len(key) == 1:
        return key.upper()
    return key


def extract_material_from_entry_dict(entry_dict: Dict) -> str:
    """
    Extracts material name from various possible fields within an LLM's output entry.
    """
    search_paths = [
        ["Material"],  # Direct "Material" key at the top level of the entry
        ["Details", "Design", "Material"],  # Nested path
        ["material"],  # Lowercase variations
        ["Details", "Design", "material"],
        ["MaterialName"],
        # Paths from your original script
        ["Design", "Material"],
        ["Composition", "Base"],
        ["Composition"]
    ]
    for path in search_paths:
        current = entry_dict
        try:
            for key_part in path:
                # Try to find key_part case-insensitively first if it's in a sub-dict
                if isinstance(current, dict):
                    found_actual_key = None
                    for actual_key_in_dict in current.keys():
                        if actual_key_in_dict.lower() == key_part.lower():
                            found_actual_key = actual_key_in_dict
                            break
                    if found_actual_key:
                        current = current[found_actual_key]
                    else:  # Key part not found
                        raise KeyError
                else:  # current is not a dict, cannot go deeper
                    raise TypeError

            if current:  # Found a value
                if isinstance(current, list):  # If it's a list, take the first element
                    current = current[0] if current else "Unknown"
                return str(current)  # Return as string
        except (KeyError, TypeError, IndexError):
            continue
    return "Unknown"