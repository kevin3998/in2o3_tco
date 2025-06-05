# flatten_extracted_data.py
import json
import pandas as pd
import argparse
import os
from typing import List, Dict, Any, Union, Optional


def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flattens a nested dictionary.
    Lists of complex objects (dictionaries) are converted to JSON strings.
    Lists of simple items (strings, numbers) are joined into a single string.
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if not v:  # Empty list
                items.append((new_key, ""))
            elif all(isinstance(i, (str, int, float, bool)) for i in v):  # List of simple items
                items.append((new_key, "; ".join(map(str, v))))
            else:  # List of complex objects (likely dicts)
                try:
                    items.append((new_key, json.dumps(v, ensure_ascii=False)))
                except TypeError:
                    items.append((new_key, str(v)))  # Fallback for non-serializable
        else:
            items.append((new_key, v))
    return dict(items)


def process_extracted_json(input_json_path: str, output_csv_path: str, selected_columns: Optional[List[str]] = None):
    """
    Processes the structured JSON output, extracts relevant data,
    flattens it, and saves it as a CSV table.

    Args:
        input_json_path (str): Path to the input JSON file from the extractor.
        output_csv_path (str): Path to save the output CSV file.
        selected_columns (Optional[List[str]]): Specific columns to include in the output.
                                               If None, all found columns will be used.
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data_entries = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return

    if not isinstance(data_entries, list):
        print(f"Error: Expected input JSON to be a list of entries. Got {type(data_entries)}")
        return

    if not data_entries:
        print("Input JSON is empty. No data to process.")
        return

    processed_rows = []
    all_found_keys = set()  # To dynamically determine all possible columns if selected_columns is None

    for entry in data_entries:
        if not isinstance(entry, dict):
            print(f"Warning: Skipping non-dictionary entry: {entry}")
            continue

        extracted_material_data = entry.get("extracted_material_data")
        if not isinstance(extracted_material_data, dict):
            print(
                f"Warning: Missing or invalid 'extracted_material_data' in entry: {entry.get('meta_source_paper', {}).get('doi', 'Unknown DOI')}")
            continue

        material_name = extracted_material_data.get("MaterialName", "UnknownMaterial")
        details = extracted_material_data.get("Details")

        row_data = {"MaterialName": material_name}

        if isinstance(details, dict):
            flattened_details = flatten_dict(details)
            row_data.update(flattened_details)
        else:
            print(f"Warning: 'Details' for {material_name} is not a dictionary or is missing.")

        processed_rows.append(row_data)
        if selected_columns is None:  # If no columns selected, collect all keys
            all_found_keys.update(row_data.keys())

    if not processed_rows:
        print("No valid material data found to process into a table.")
        return

    # Create DataFrame
    df = pd.DataFrame(processed_rows)

    # Order and select columns
    if selected_columns:
        # Ensure 'MaterialName' is first if present in selected_columns, or add it
        final_columns = []
        if "MaterialName" in selected_columns:
            final_columns.append("MaterialName")
            final_columns.extend([col for col in selected_columns if col != "MaterialName"])
        else:
            final_columns = ["MaterialName"] + selected_columns

        # Include only existing columns from the dataframe to avoid errors
        # and fill missing ones with a placeholder (NaN by default with reindex)
        existing_final_columns = [col for col in final_columns if col in df.columns]
        missing_selected_cols = [col for col in final_columns if col not in df.columns]
        if missing_selected_cols:
            print(
                f"Warning: The following selected columns were not found in the data and will be omitted or empty: {missing_selected_cols}")

        df_ordered = pd.DataFrame()  # Create an empty DataFrame
        for col in existing_final_columns:  # Add existing columns
            df_ordered[col] = df[col]
        for col in missing_selected_cols:  # Add missing selected columns as empty
            if col not in df_ordered.columns:  # Check if not already added (e.g. if MaterialName was missing but selected)
                df_ordered[col] = pd.NA

        df = df_ordered.reindex(columns=final_columns)  # Ensure final order and presence of all selected columns

    elif all_found_keys:  # Use all dynamically found keys
        # Bring MaterialName to the front if it exists
        ordered_keys = sorted(list(all_found_keys))
        if "MaterialName" in ordered_keys:
            ordered_keys.insert(0, ordered_keys.pop(ordered_keys.index("MaterialName")))
        df = df.reindex(columns=ordered_keys)

    # Save to CSV
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compatibility with Unicode
        print(f"Successfully created table at: {output_csv_path}")
        print(f"Table dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    # Optionally print some info
    if not df.empty:
        print("\nFirst 5 rows of the table:")
        print(df.head().to_string())
        # print("\nTable columns:")
        # print(df.columns.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert structured JSON output to a CSV table.")
    parser.add_argument("input_json", help="Path to the input structured JSON file (output of the extractor).")
    parser.add_argument("output_csv", help="Path to save the output CSV file.")
    parser.add_argument(
        "--columns",
        nargs='*',  # Allows zero or more arguments for --columns
        help="Space-separated list of selected column names to include in the output. "
             "Prefix nested keys with parent keys, e.g., Design_HostMaterial Performance_ElectricalProperties_Resistivity. "
             "If not provided, all found keys will be used."
    )
    args = parser.parse_args()

    # --- DEFINE YOUR DESIRED COLUMNS HERE IF NOT USING --columns ARGUMENT, OR AS A DEFAULT ---
    # This list should ideally be generated based on your Pydantic schemas for consistency
    # These are examples based on TCO and Membrane schemas we discussed.
    # Customize this list extensively for your specific needs.
    DEFAULT_SELECTED_COLUMNS = [
        "MaterialName",
        # Common Design fields
        "Design_HostMaterial",  # TCO
        "Design_PrimaryDopant_element",  # TCO
        "Design_PrimaryDopant_concentration_text",  # TCO
        "Design_CoDopants",  # TCO (will be JSON string)
        "Design_BasePolymer",  # Membrane (will be JSON string of MembranePolymerSchema)
        "Design_Solvents",  # Membrane (will be JSON string of MembraneSolventSchema)
        "Design_Additives",  # Membrane (will be JSON string of MembraneAdditiveSchema)

        # Common Fabrication fields
        "Fabrication_DepositionMethod",  # TCO
        "Fabrication_PrimaryMethod",  # Membrane
        "Fabrication_SubstrateMaterial",
        "Fabrication_AnnealingConditions_Temperature",  # TCO (AnnealingConditions is an object)
        "Fabrication_AnnealingConditions_Atmosphere",  # TCO
        "Fabrication_CoagulationBathDetails_Temperature",  # Membrane
        "Fabrication_FilmThicknessText",

        # Common Performance fields
        "Performance_ElectricalProperties_Resistivity",  # TCO
        "Performance_ElectricalProperties_SheetResistance",  # TCO
        "Performance_ElectricalProperties_CarrierConcentration",  # TCO
        "Performance_ElectricalProperties_HallMobility",  # TCO
        "Performance_OpticalProperties_AverageTransmittance",  # TCO
        "Performance_OpticalProperties_OpticalBandGapText",  # TCO
        "Performance_OtherPerformanceMetrics_WorkFunctionText",  # TCO
        "Performance_OtherPerformanceMetrics_FigureOfMeritValue",  # TCO
        "Performance_LiquidTransportProperties_PureWaterFlux",  # Membrane
        "Performance_LiquidTransportProperties_Rejections",  # Membrane (will be JSON string)
        "Performance_GasTransportProperties_Permeances",  # Membrane (will be JSON string)
        "Performance_GasTransportProperties_Selectivities",  # Membrane (will be JSON string)
        "Performance_StructuralPhysicalProperties_PoreSizeText",  # Membrane
        "Performance_StructuralPhysicalProperties_ContactAngleText",  # Membrane

        # Common Application fields
        "Application_PotentialApplicationArea",  # TCO & Membrane (key might differ slightly based on schema alias)
        "Application_DevicePerformance",  # TCO
        "Application_AchievedPerformanceInApplication"  # Membrane
    ]

    selected_cols = args.columns
    if args.columns is None:  # If --columns flag was not used at all
        # You can choose to use all dynamic columns or default to a predefined set
        # Option 1: Use all dynamically found columns (can be very wide)
        # selected_cols = None
        # Option 2: Use a predefined default list (recommended for consistency)
        print(
            f"No specific columns provided via --columns argument. Using a predefined set of {len(DEFAULT_SELECTED_COLUMNS)} columns.")
        print("To customize columns, use the --columns argument or modify DEFAULT_SELECTED_COLUMNS in the script.")
        selected_cols = DEFAULT_SELECTED_COLUMNS
    elif not args.columns:  # If --columns flag was used but no column names were provided after it
        print("Warning: --columns flag was used but no column names were provided. All found columns will be used.")
        selected_cols = None

    process_extracted_json(args.input_json, args.output_csv, selected_columns=selected_cols)