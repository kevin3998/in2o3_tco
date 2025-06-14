# 5_create_finetune_set.py
import json
import os
import argparse
import logging
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- System Prompt Definition ---
# 您可以根据需要修改这个系统提示，它将定义微调后模型的基本行为。
SYSTEM_PROMPT = "You are an expert AI assistant specializing in materials science. Your task is to accurately and directly answer the user's questions based on scientific data. Do not ask questions, provide answers."


def convert_to_finetune_format(input_path: str, output_path: str, system_prompt: str):
    """
    Converts the intermediate QA pairs file to the final fine-tuning format (JSONL).

    Args:
        input_path (str): Path to the input .jsonl file generated by 4_qa_generator.py.
        output_path (str): Path to save the final training-ready .jsonl file.
        system_prompt (str): The system prompt to include in every training example.
    """
    logging.info(f"Starting conversion from '{input_path}' to fine-tuning format.")

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:

            lines = infile.readlines()
            processed_count = 0

            for line in tqdm(lines, desc="Converting QA pairs"):
                try:
                    qa_pair = json.loads(line)
                    question = qa_pair.get("question")
                    answer = qa_pair.get("answer")

                    if question and answer:
                        # Construct the fine-tuning record in messages format
                        finetune_record = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                        }
                        outfile.write(json.dumps(finetune_record, ensure_ascii=False) + "\n")
                        processed_count += 1
                    else:
                        logging.warning(f"Skipping malformed QA pair: {line.strip()}")

                except json.JSONDecodeError:
                    logging.warning(f"Skipping line that is not valid JSON: {line.strip()}")
                    continue

        logging.info(f"✅ Conversion complete. Successfully processed {processed_count} QA pairs.")
        logging.info(f"Fine-tuning dataset saved to: {output_path}")

    except FileNotFoundError:
        logging.error(f"Input file not found at '{input_path}'. Please check the path.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during conversion: {e}", exc_info=True)


def main():
    """Main function to run the conversion script."""
    # =================================================================================
    # === Configuration Block for PyCharm/IDE Run ===
    # Set to True to use the settings below.
    # Set to False to use command-line arguments when running from the terminal.
    IDE_RUN_CONFIG = True
    # =================================================================================

    args = None
    if IDE_RUN_CONFIG:
        logging.info("--- RUNNING IN IDE MODE: Using parameters defined in the script. ---")

        class Args:
            pass

        args = Args()

        # --- SET YOUR PARAMETERS FOR IDE RUNS HERE ---
        # 确保这个输入文件是上一步生成的QA对文件
        args.input_file = "../data/generated_qa_pairs/finetune_training_set_mixed_final.jsonl"
        # 这是最终可以直接用于训练的文件
        args.output_file = "../data/finetune_sets/finetune_training_set_mixed_final.jsonl"
        args.system_prompt = SYSTEM_PROMPT
        # ---------------------------------------------
    else:
        logging.info("--- RUNNING IN COMMAND-LINE MODE: Parsing arguments from terminal. ---")
        parser = argparse.ArgumentParser(description="Convert QA pairs to the format required for fine-tuning.")
        parser.add_argument("--input_file", type=str, required=True, help="Path to the qa_pairs.jsonl file.")
        parser.add_argument("--output_file", type=str, required=True, help="Path to save the final training_set.jsonl.")
        parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT, help="The system prompt for the model.")
        args = parser.parse_args()

    convert_to_finetune_format(args.input_file, args.output_file, args.system_prompt)


if __name__ == "__main__":
    main()