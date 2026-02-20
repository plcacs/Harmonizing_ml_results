from dotenv import load_dotenv
import os
import json
import time
import tiktoken
from openai import OpenAI
from pathlib import Path

load_dotenv()

client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "Files_not_for_root_directories")

PERCENT_FOLDERS = [
    "ten_percent_typed",
    "twenty_percent_typed",
    "thirty_percent_typed",
    "forty_percent_typed",
    "fifty_percent_typed",
    "sixty_percent_typed",
    "seventy_percent_typed",
    "eighty_percent_typed",
    "ninety_percent_typed",
]

# --- CONFIGURE THESE ---
BATCH_SIZE = 100  # Number of files to process per folder per run
# -----------------------


def get_paths(folder_name: str):
    """Return input dir, output dir, and log file paths for a given percent folder."""
    input_dir = os.path.join(BASE_DIR, folder_name)
    output_dir = os.path.join(BASE_DIR, "o3_mini_outputs", f"o3_mini_{folder_name}_output")
    processed_log = os.path.join(LOGS_DIR, f"processed_files_o3_mini_{folder_name}.txt")
    timing_log = os.path.join(LOGS_DIR, f"o3_mini_timings_{folder_name}.json")
    unprocessed_log = os.path.join(LOGS_DIR, f"unprocessed_files_o3_mini_{folder_name}.txt")
    return input_dir, output_dir, processed_log, timing_log, unprocessed_log


def get_token_count(text: str, model: str = "o3-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def log_timing(timing_log: str, file_path, duration):
    file_path_str = str(file_path)
    log_entry = {"file": file_path_str, "time_taken": duration}

    data = []
    if os.path.exists(timing_log):
        try:
            with open(timing_log, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = []

    data.append(log_entry)
    with open(timing_log, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_processed_files(processed_log: str) -> set:
    if os.path.exists(processed_log):
        with open(processed_log, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def generate_type_annotated_code(code: str):
    prompt = (
        f"Here is a partially type-annotated Python program:\n\n{code}\n\n"
        "Complete the missing type annotations while keeping existing annotations unchanged. "
        "Output only the fully annotated Python code. No explanation needed."
    )
    get_token_count(prompt, model="o3-mini")

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
            return response.choices[0].message, 1, end_time - start_time
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return code, 2, 0
            elif "rate_limit_exceeded" in error_msg:
                print(f"Rate limit exceeded. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                wait_time += 30
            else:
                print(f"Error generating type-annotated code: {e}")
                return code, 2, 0
    print("Max retries reached. Skipping file.")
    return code, 2, 0


def process_file(file_path: Path, output_dir: str, processed_log: str, timing_log: str, unprocessed_log: str):
    file_key = str(file_path)
    processed_files = load_processed_files(processed_log)
    if file_key in processed_files:
        print(f"  Skipping {file_path.name}, already processed.")
        return

    print(f"  Processing: {file_path.name}")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"  Skipping {file_path.name} due to read error: {e}")
        return

    modified_code, is_success, api_duration = generate_type_annotated_code(code)
    log_timing(timing_log, file_path, api_duration)

    if is_success != 1:
        print(f"  Skipping {file_path.name} due to generation error.")
        with open(unprocessed_log, "a", encoding="utf-8") as f:
            f.write(file_key + "\n")
        return

    content = modified_code.content if hasattr(modified_code, "content") else modified_code
    if isinstance(content, str):
        if "```python" in content:
            try:
                code_block = content.split("```python\n")[1].split("```")[0]
            except IndexError:
                code_block = content
        else:
            code_block = content
    else:
        print(f"  Skipping {file_path.name} due to unexpected response format")
        return

    os.makedirs(output_dir, exist_ok=True)
    new_file_path = os.path.join(output_dir, file_path.name)
    try:
        with open(new_file_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(code_block)
    except (UnicodeDecodeError, IOError) as e:
        print(f"  Skipping {file_path.name} due to write error: {e}")
        return

    print(f"  Done: {file_path.name}")
    with open(processed_log, "a", encoding="utf-8") as f:
        f.write(file_key + "\n")


def process_folder(folder_name: str, batch_size: int):
    input_dir, output_dir, processed_log, timing_log, unprocessed_log = get_paths(folder_name)

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist. Skipping.")
        return

    python_files = sorted(Path(input_dir).glob("*.py"))
    if not python_files:
        print(f"No Python files found in {input_dir}. Skipping.")
        return

    processed_files = load_processed_files(processed_log)
    remaining = [f for f in python_files if str(f) not in processed_files]

    if not remaining:
        print(f"[{folder_name}] All {len(python_files)} files already processed.")
        return

    batch = remaining[:batch_size]
    print(f"\n[{folder_name}] Total: {len(python_files)} | Already done: {len(python_files) - len(remaining)} | "
          f"Remaining: {len(remaining)} | This batch: {len(batch)}")

    for i, file_path in enumerate(batch, 1):
        process_file(file_path, output_dir, processed_log, timing_log, unprocessed_log)
        print(f"  [{folder_name}] Batch progress: {i}/{len(batch)}")
        if i < len(batch):
            time.sleep(5)


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    for folder_name in PERCENT_FOLDERS:
        process_folder(folder_name, BATCH_SIZE)
        time.sleep(10)

    print("\n=== Run complete ===")


if __name__ == "__main__":
    main()
