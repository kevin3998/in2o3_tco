# qa_generator_CoT.py
import json
import time
import os
import regex as re
import argparse
import logging
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)


# --- 2. LLM Client Initialization ---
def get_llm_client() -> OpenAI:
    """Initializes and returns the OpenAI client, loading credentials from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables (.env file).")
        raise ValueError("OPENAI_API_KEY not found.")

    return OpenAI(api_key=api_key, base_url=base_url)


# --- 3. Checkpoint Management ---
def save_checkpoint(checkpoint_path: str, data: dict):
    """Saves progress to a checkpoint file."""
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.debug(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")


def load_checkpoint(checkpoint_path: str) -> dict:
    """Loads progress from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load checkpoint {checkpoint_path}: {e}. Starting fresh.")
    return {"processed_entries": 0, "qa_pairs": []}


# --- 4. Core CoT QA Generation Logic ---

# --- STAGE 1: Complex Question Generation ---
COMPLEX_QUESTION_PROMPT_TEMPLATE = """
You are a senior materials science researcher with deep expertise in analyzing experimental data.
Your task is to generate complex, insightful questions based on structured data from a scientific paper.

**Context:**
The following JSON data describes a material, '{material_name}', and its properties or synthesis conditions.
{material_data_json}

**Task:**
Based *only* on the data provided above, generate {num_questions} distinct and complex questions. These questions should:
- Require multi-step reasoning, inference, or comparison of different data points.
- Go beyond simple fact retrieval (e.g., avoid "What is the deposition temperature?").
- Be fully answerable using *only* the provided data.

**Good Example Question:** "Given the observed changes in carrier concentration and mobility at different annealing temperatures, what is the likely dominant scattering mechanism in the {material_name} thin films?"

**Output Format:**
Respond with a single, valid JSON array of strings. Each string in the array is one question.
Do NOT include any other text, explanations, or markdown formatting like ```json.

**Generated JSON Array:**
"""

# --- STAGE 2: Chain-of-Thought Answer Generation ---
CHAIN_OF_THOUGHT_ANSWER_PROMPT_TEMPLATE = """
You are a meticulous AI assistant trained to provide detailed, evidence-based answers in materials science.
Your answer must follow a clear Chain-of-Thought process and be based *exclusively* on the provided data.

**Available Data for material '{material_name}':**
{material_data_json}

**Question to Answer:**
"{complex_question}"

**Task:**
Generate a comprehensive answer by following the structure below. Your entire response should be a single block of text.

1.  **Thought Process:** Clearly outline the logical steps required to answer the question using the available data.
2.  **Analysis and Evidence:** Execute your plan. Go through each step, extracting and citing the specific data points from the JSON context that support your analysis.
3.  **Conclusion:** Based on the evidence and analysis, provide a final, concise conclusion that directly answers the question.

**Answer:**
"""


def _format_data_for_prompt(data: Dict) -> str:
    """Converts the details dictionary into a nicely formatted JSON string for the prompt."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Generic function to call the LLM and handle basic response validation."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            finish_reason = response.choices[0].finish_reason if response.choices else "unknown reason"
            logging.warning(f"LLM returned no content. Finish reason: {finish_reason}.")
            return ""
    except Exception as e:
        logging.error(f"LLM API call failed: {e}")
        return ""


def process_structured_data_for_cot_qa(
        client: OpenAI,
        model_name: str,
        structured_data: List[Dict],
        checkpoint_path: str
) -> List[Dict[str, Any]]:
    """
    Processes structured data to generate Chain-of-Thought (CoT) QA pairs.
    This involves a two-stage LLM call process:
    1. Generate complex questions for a material.
    2. Generate detailed, reasoned answers for each question.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    all_qa_pairs = checkpoint.get("qa_pairs", [])
    start_index = checkpoint.get("processed_entries", 0)

    total_entries = len(structured_data)
    logging.info(
        f"Starting CoT QA generation. Total entries: {total_entries}. Already processed: {start_index}.")

    for i, entry in enumerate(structured_data[start_index:], start=start_index):
        material_name = "this material"  # Default name
        try:
            material_data = entry.get("extracted_material_data", {})
            material_name = material_data.get("MaterialName", "this material")
            details = material_data.get("Details", {})

            if not details:
                logging.info(f"No 'Details' data for material '{material_name}'. Skipping.")
                continue

            material_data_json_str = _format_data_for_prompt(details)

            # --- STAGE 1: Generate Complex Questions ---
            logging.info(f"Stage 1: Generating complex questions for '{material_name}'...")
            question_prompt = COMPLEX_QUESTION_PROMPT_TEMPLATE.format(
                material_name=material_name,
                material_data_json=material_data_json_str,
                num_questions=3
            )

            question_response_str = call_llm(client, model_name, question_prompt, max_tokens=2048, temperature=0.5)
            if not question_response_str:
                logging.warning(f"Failed to generate questions for '{material_name}'. Skipping.")
                continue

            try:
                # The response might be inside a markdown block, so we extract it.
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', question_response_str)
                if json_match:
                    clean_json_str = json_match.group(1)
                else:
                    clean_json_str = question_response_str

                complex_questions = json.loads(clean_json_str)

                if not isinstance(complex_questions, list) or not all(isinstance(q, str) for q in complex_questions):
                    raise TypeError("LLM response is not a list of strings.")

            except (json.JSONDecodeError, TypeError) as e:
                logging.error(
                    f"Failed to parse question list from LLM for '{material_name}'. Error: {e}. Response:\n{question_response_str}")
                continue

            logging.info(f"Generated {len(complex_questions)} questions for '{material_name}'.")

            # --- STAGE 2: Generate Chain-of-Thought Answers ---
            for question in complex_questions:
                logging.info(f"  Stage 2: Generating CoT answer for question: '{question[:80]}...'")
                answer_prompt = CHAIN_OF_THOUGHT_ANSWER_PROMPT_TEMPLATE.format(
                    material_name=material_name,
                    material_data_json=material_data_json_str,
                    complex_question=question
                )

                cot_answer = call_llm(client, model_name, answer_prompt, max_tokens=4096, temperature=0.2)
                if not cot_answer:
                    logging.warning(
                        f"    Failed to generate a CoT answer for '{material_name}'. Skipping this question.")
                    continue

                qa_pair = {
                    "source_doi": entry.get("meta_source_paper", {}).get("doi"),
                    "material_name": material_name,
                    "question": question,
                    "answer": cot_answer
                }
                all_qa_pairs.append(qa_pair)
                time.sleep(1)

            progress = (i + 1) / total_entries * 100
            logging.info(
                f"Progress: {progress:.2f}% ({i + 1}/{total_entries}) - Finished processing material '{material_name}'")

            if (i + 1) % 2 == 0:
                save_checkpoint(checkpoint_path, {"processed_entries": i + 1, "qa_pairs": all_qa_pairs})

        except Exception as e:
            logging.error(f"An unexpected error occurred processing entry {i} ('{material_name}'): {e}", exc_info=True)
            continue

    save_checkpoint(checkpoint_path, {"processed_entries": total_entries, "qa_pairs": all_qa_pairs})
    return all_qa_pairs


# --- 5. Main Execution Block ---
def main():
    """Main function to run the CoT QA pair generation pipeline."""
    # --- MODIFICATION: Set to True to run with hardcoded paths, False to use command-line args ---
    IDE_RUN_CONFIG = True
    args = None

    if IDE_RUN_CONFIG:
        logging.info("--- RUNNING IN IDE MODE: Using parameters defined in the script. ---")

        # Simple class to mimic argparse's namespace object
        class Args:
            pass

        args = Args()
        # Define hardcoded paths for IDE execution
        args.input_file = "../data/extracted_json/pydantic/structured_info.json"
        args.output_file = "../data/generated_qa_pairs/qa_pairs_CoT.jsonl"
        args.checkpoint_file = "data/generated_qa_pairs/checkpoint_qa_CoT.json"
        args.model_name = "DeepSeek-R1-671B"  # or another powerful model suitable for CoT

    else:
        logging.info("--- RUNNING IN COMMAND-LINE MODE: Parsing arguments from terminal. ---")
        parser = argparse.ArgumentParser(
            description="Generate Chain-of-Thought (CoT) Question-Answer pairs from structured JSON."
        )
        parser.add_argument("--input_file", type=str, default="data/extracted_json/pydantic/structured_info.json",
                            help="Path to the structured JSON input file.")
        parser.add_argument("--output_file", type=str, default="data/generated_qa_pairs/qa_pairs_CoT.jsonl",
                            help="Path to save the generated CoT QA pairs (JSONL).")
        parser.add_argument("--checkpoint_file", type=str, default="data/generated_qa_pairs/checkpoint_qa_CoT.json",
                            help="Path for the checkpoint file.")
        parser.add_argument("--model_name", type=str, default="DeepSeek-R1-671B", help="LLM to use for generation.")
        args = parser.parse_args()

    try:
        client = get_llm_client()
        logging.info(f"Loading structured data from: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            structured_data_list = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from input file: {args.input_file}")
        return
    except Exception as e:
        logging.error(f"An error occurred during setup: {e}")
        return

    if not isinstance(structured_data_list, list):
        logging.error(f"Error: Expected input file {args.input_file} to contain a JSON list.")
        return

    final_qa_pairs = process_structured_data_for_cot_qa(
        client,
        args.model_name,
        structured_data_list,
        args.checkpoint_file
    )

    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for qa_pair in final_qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
        logging.info(
            f"\n✅ CoT QA pair generation complete. {len(final_qa_pairs)} pairs saved to {args.output_file} in JSONL format."
        )
    except Exception as e:
        logging.error(f"Error writing output file {args.output_file}: {e}")

    # Clean up the checkpoint file on successful completion
    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)
        logging.info(f"Checkpoint file {args.checkpoint_file} removed.")


if __name__ == "__main__":
    main()