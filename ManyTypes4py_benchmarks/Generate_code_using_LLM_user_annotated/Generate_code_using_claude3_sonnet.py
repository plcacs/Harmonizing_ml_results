import os
import json
import time
import tiktoken
import anthropic
import hashlib
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from environment
PROCESSED_FILES_LOG = "ManyTypes4py_benchmarks/Files_not_for_root_directories/processed_files_claude3_sonnet_user_annotated.txt"
JSON_FILE = "ManyTypes4py_benchmarks/Files_not_for_root_directories/original_files_grouped_paths.json"
OUTPUT_DIR = "ManyTypes4py_benchmarks/claude3_sonnet_user_annotated"
TIMING_LOG = "ManyTypes4py_benchmarks/Files_not_for_root_directories/claude3_sonnet_model_timings_user_annotated.json"
UNPROCESSED_FILES = "ManyTypes4py_benchmarks/Files_not_for_root_directories/unprocessed_files_claude3_sonnet_user_annotated.txt"
UNEXPECTED_FORMAT_LOG = "ManyTypes4py_benchmarks/Files_not_for_root_directories/unexpected_format_claude3_sonnet.txt"

MODEL_NAME = "claude-3-7-sonnet-latest"  # Anthropic Claude 3 Sonnet


def get_token_count(text: str, model: str = MODEL_NAME) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def log_timing(file_path, duration):
    log_entry = {"file": file_path, "time_taken": duration}
    if os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
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
    prompt_tokens = get_token_count(prompt, model=MODEL_NAME)
    # Dynamically set max_tokens: twice the code tokens, capped at 64,000, and ensure total context <= 200,000
    max_tokens = min(64000, (prompt_tokens * 2)+1000)
  
    token_count = prompt_tokens
    max_retries = 3
    wait_time = 60
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=max_tokens,  # dynamically set value
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stream=True,  # Enable streaming to avoid timeout warnings
            )
            # Handle streaming response
            content_parts = []
            for chunk in response:
                if chunk.type == "content_block_delta":
                    content_parts.append(chunk.delta.text)
            
            content = "".join(content_parts)
            end_time = time.time()
            log_timing("claude3_sonnet_annotation", end_time - start_time)
            return content, 1
        except Exception as e:
            error_msg = str(e)
            print(f"Error code: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return code, 2
            elif "rate_limit_exceeded" in error_msg:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                wait_time += 30
            else:
                print(f"Error generating type-annotated code: {e}")
                return code, 2
    print("Max retries reached. Skipping file.")
    return code, 2

def extract_code_from_response(content: str) -> str:
    """Extract Python code from the response, handling various formats."""
    # Try to find code blocks
    if '```python' in content:
        try:
            # Extract code between ```python and ```
            code_block = content.split('```python\n')[1].split('```')[0]
            return code_block
        except IndexError:
            pass
    
    if '```' in content:
        try:
            # Try without 'python' specifier
            code_block = content.split('```\n')[1].split('```')[0]
            return code_block
        except IndexError:
            pass
    
    # If no code blocks found, return the content as-is
    return content

def process_file(file_path, grouped_id):
    processed_files = load_processed_files()
    if file_path in processed_files:
        print(f"Skipping {file_path}, already processed.")
        return
    print(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
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
            f.write(file_path + "\n")
        return
    
    # Extract code from response
    code_block = extract_code_from_response(modified_code)
    
    # If extraction failed, log the unexpected format
    if not code_block or code_block == modified_code:
        print(f"Skipping file {file_path} due to unexpected format")
        with open(UNEXPECTED_FORMAT_LOG, "a", encoding="utf-8") as f:
            f.write(f"File: {file_path}\nResponse:\n{modified_code}\n{'='*40}\n")
        return
    
    new_file_path = os.path.join(OUTPUT_DIR, grouped_id)
    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)
    filename = os.path.basename(file_path)
    new_file_path = os.path.join(new_file_path, filename)
    try:
        with open(new_file_path, 'w', encoding='utf-8', errors='ignore') as file:
            file.write(code_block)
    except (UnicodeEncodeError, IOError) as e:
        print(f"Skipping {file_path} due to write error: {e}")
        return
    print(f"Successfully processed: {file_path}")
    with open(PROCESSED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")

def process_files_from_json():
    processed_files = load_processed_files()
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            file_map = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    all_files = []
    for id_ in range(1, 19):
        grouped_id = str(id_)
        all_files.extend(file_map.get(grouped_id, []))
    files_to_process = [f for f in all_files if f not in processed_files]
    total_to_process = len(files_to_process)
    processed_count = 0
    left_count = total_to_process
    for id_ in range(1, 19):
        grouped_id = str(id_)
        for file_path in file_map[grouped_id]:
            if file_path in processed_files:
                print(f"Skipping already processed file: {file_path}")
                continue
            process_file(file_path, grouped_id)
            processed_count += 1
            left_count -= 1
            print(f"Processed: {processed_count}, Left: {left_count}")
            time.sleep(5)
        time.sleep(600)

if __name__ == "__main__":
    process_files_from_json() 