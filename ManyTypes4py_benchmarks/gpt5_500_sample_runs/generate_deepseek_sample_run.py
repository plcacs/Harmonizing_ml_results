"""
Run DeepSeek type annotation on the 500 selected sample files.

Usage:
    python generate_deepseek_sample_run.py <run_number>

    run_number: 2, 3, or 4
"""

import json
import os
import sys
import time

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

REPO_ROOT = os.path.dirname(PARENT_DIR)
load_dotenv(os.path.join(REPO_ROOT, ".env"))

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(
        "DEEPSEEK_API_KEY not set. Create a .env file in the repo root."
    )

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

MODEL_NAME = "deepseek-chat"

SELECTED_FILES_JSON = os.path.join(SCRIPT_DIR, "selected_500_files.json")
GROUPED_JSON = os.path.join(
    PARENT_DIR, "Files_not_for_root_directories", "grouped_file_paths.json"
)


def get_run_paths(run_number):
    run_name = f"deepseek_{run_number}_run"
    logs_dir = os.path.join(PARENT_DIR, "Files_not_for_root_directories")
    return {
        "output_dir": os.path.join(PARENT_DIR, run_name),
        "processed_log": os.path.join(logs_dir, f"processed_files_{run_name}.txt"),
        "timing_log": os.path.join(logs_dir, f"{run_name}_model_timings.json"),
        "unprocessed_log": os.path.join(logs_dir, f"unprocessed_files_{run_name}.txt"),
    }


def get_token_count(text, model=MODEL_NAME):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def log_timing(file_path, duration, timing_log):
    if os.path.exists(timing_log):
        with open(timing_log, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append({"file": file_path, "time_taken": duration})
    with open(timing_log, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_processed_files(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def generate_type_annotated_code(code):
    prompt = f"""Here is a Python program:

{code}

Add Python type annotations to the existing code.

Rules:
1. Only add type annotations to function parameters and return types.
2. Do not modify the program logic or existing code.
3. Do not add explanations, comments, or extra text.
4. Output only the complete annotated Python program.

Return only valid Python code.
Use standard Python type hints (PEP 484)."""
    max_retries = 3
    wait_time = 60
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a python programming expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            duration = time.time() - start_time
            return response.choices[0].message, True, duration
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return code, False, 0
            elif "rate_limit_exceeded" in error_msg:
                print(
                    f"Rate limit. Retrying in {wait_time}s "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                wait_time += 30
            else:
                return code, False, 0
    print("Max retries reached. Skipping.")
    return code, False, 0


def process_files(run_number):
    paths = get_run_paths(run_number)

    with open(SELECTED_FILES_JSON, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
    selected_filenames = set(selected_data["files"])

    with open(GROUPED_JSON, "r", encoding="utf-8") as f:
        file_map = json.load(f)

    processed_files = load_processed_files(paths["processed_log"])

    files_to_run = []
    for group_id in sorted(file_map.keys(), key=int):
        for file_path in file_map[group_id]:
            basename = os.path.basename(file_path)
            if basename in selected_filenames:
                files_to_run.append((group_id, file_path))

    total = len(files_to_run)
    already_done = sum(1 for _, fp in files_to_run if fp in processed_files)
    remaining = total - already_done
    print(
        f"DeepSeek run {run_number}: {total} selected files, "
        f"{already_done} already done, {remaining} remaining"
    )

    processed_count = 0
    for group_id, file_path in files_to_run:
        if file_path in processed_files:
            continue

        print(f"Processing: {file_path}")
        full_path = os.path.join(PARENT_DIR, file_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except (UnicodeDecodeError, IOError) as e:
            print(f"Read error, skipping: {e}")
            continue

        modified_code, success, duration = generate_type_annotated_code(code)
        log_timing(file_path, duration, paths["timing_log"])

        if not success:
            print(f"Skipping {file_path} — generation failed")
            with open(paths["unprocessed_log"], "a", encoding="utf-8") as f:
                f.write(file_path + "\n")
            continue

        content = (
            modified_code.content
            if hasattr(modified_code, "content")
            else modified_code
        )

        out_dir = os.path.join(paths["output_dir"], group_id)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, os.path.basename(file_path))
        try:
            with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(content)
        except (UnicodeEncodeError, IOError) as e:
            print(f"Write error, skipping: {e}")
            continue

        with open(paths["processed_log"], "a", encoding="utf-8") as f:
            f.write(file_path + "\n")

        processed_count += 1
        print(f"Done [{processed_count}/{remaining}]: {file_path}")
        time.sleep(5)


if __name__ == "__main__":
    valid_runs = {"2", "3", "4"}
    if len(sys.argv) != 2 or sys.argv[1] not in valid_runs:
        print("Usage: python generate_deepseek_sample_run.py <run_number>")
        print("  run_number: 2, 3, or 4")
        sys.exit(1)

    process_files(int(sys.argv[1]))
