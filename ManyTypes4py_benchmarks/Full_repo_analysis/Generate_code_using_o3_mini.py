from dotenv import load_dotenv
import os
import json
import time
import tiktoken
from openai import OpenAI
from pathlib import Path
import hashlib

load_dotenv()  # Load environment variables from .env

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PROCESSED_FILES_LOG = "processed_files_o3_mini_large.txt"
INPUT_DIR = "untyped_version_large"
OUTPUT_DIR = "o3_mini_output_large"
TIMING_LOG = "o3_mini_model_timings_large.json"
UNPROCESSED_FILES = "unprocessed_files_o3_mini_large.txt"


def get_token_count(text: str, model: str = "o3-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def log_timing(file_path, duration):
    # Convert Path object to string for JSON serialization
    file_path_str = str(file_path) if hasattr(file_path, "__str__") else file_path
    log_entry = {"file": file_path_str, "time_taken": duration}

    # Load existing data with error handling for corrupted JSON
    data = []
    if os.path.exists(TIMING_LOG):
        try:
            with open(TIMING_LOG, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Corrupted timing log file. Starting fresh. Error: {e}")
            data = []

    data.append(log_entry)
    with open(TIMING_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def generate_type_annotated_code(code: str) -> str:
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the annotated Python code. No Explanation needed."
    token_count = get_token_count(prompt, model="o3-mini")

    max_retries = 3
    wait_time = 60
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            end_time = time.time()
            log_timing("o3mini_annotation", end_time - start_time)
            return response.choices[0].message, 1
        except Exception as e:
            error_msg = str(e)
            print(f"Error code: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return code, 2
            elif "rate_limit_exceeded" in error_msg:
                print(
                    f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})"
                )
                time.sleep(wait_time)
                wait_time += 30
            else:
                print(f"Error generating type-annotated code: {e}")
                return code, 2
    print("Max retries reached. Skipping file.")

    return code, 2


def process_file(file_path):
    processed_files = load_processed_files()
    if file_path in processed_files:
        print(f"Skipping {file_path}, already processed.")
        return
    print(f"Processing file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            code = file.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"Skipping {file_path} due to read error: {e}")
        return
    start_time = time.time()
    modified_code, isSuccess = generate_type_annotated_code(code)

    end_time = time.time()
    log_timing(file_path, end_time - start_time)
    if isSuccess == 0:
        print(f"Skipping file {file_path} due to generation error or token limit.")
        with open(UNPROCESSED_FILES, "a", encoding="utf-8") as f:
            f.write(str(file_path) + "\n")
        return
    content = (
        modified_code.content if hasattr(modified_code, "content") else modified_code
    )

    # Handle o3-mini response format - it returns code directly without markdown blocks
    if isinstance(content, str):
        # Check if content contains markdown code blocks
        if "```python" in content:
            try:
                code_block = content.split("```python\n")[1].split("```")[0]
            except IndexError:
                # If markdown parsing fails, use the content as-is
                code_block = content
        else:
            # o3-mini returns code directly, use as-is
            code_block = content
    else:
        print(f"Skipping file {file_path} due to unexpected response format")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Keep the original filename since files in untyped_version already have correct naming
    filename = file_path.name

    new_file_path = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(new_file_path, "w", encoding="utf-8", errors="ignore") as file:
            file.write(code_block)
    except (UnicodeDecodeError, IOError) as e:
        print(f"Skipping {file_path} due to write error: {e}")
        return
    print(f"Successfully processed: {file_path}")
    with open(PROCESSED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(str(file_path) + "\n")


def process_all_renamed_files():
    """Process all Python files in the renamed benchmarks directory"""
    if not os.path.exists(INPUT_DIR):
        print(
            f"Input directory {INPUT_DIR} does not exist. Please run rename_functions.py first."
        )
        return

    # Get all Python files from the untyped_version directory (including subdirectories)
    python_files = list(Path(INPUT_DIR).rglob("*.py"))

    if not python_files:
        print(f"No Python files found in {INPUT_DIR}")
        return

    print(f"Found {len(python_files)} Python files to process")

    processed_files = load_processed_files()
    files_to_process = [f for f in python_files if str(f) not in processed_files]

    total_to_process = len(files_to_process)
    processed_count = 0

    print(f"Total files to process: {total_to_process}")

    for file_path in files_to_process:
        process_file(file_path)
        processed_count += 1
        print(f"Processed: {processed_count}/{total_to_process}")
        time.sleep(5)  # Rate limiting

    print(f"Completed! Processed {processed_count} files")


if __name__ == "__main__":
    process_all_renamed_files()
