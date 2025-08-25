import os
import json
import time
import tiktoken
import anthropic
import hashlib
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from environment
PROCESSED_FILES_LOG = "LLM_Gen_Files/processed_files_claude3_sonnet_2nd_run.txt"

OUTPUT_DIR = "claude3_sonnet_2nd_run"
TIMING_LOG = "LLM_Gen_Files/claude3_sonnet_model_timings_2nd_run.json"
UNPROCESSED_FILES = "LLM_Gen_Files/unprocessed_files_claude3_sonnet_2nd_run.txt"
INPUT_DIR = "untyped_benchmarks"
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

def ensure_parent_directory_exists(target_path: str) -> None:
    """Create parent directory for a file path if it does not exist."""
    parent_dir = os.path.dirname(target_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

def extract_code_from_content(content: str) -> str | None:
    """Try to extract a Python code block from model content with fallbacks.

    Returns the extracted code string or None if nothing reasonable was found.
    """
    # Common fenced patterns to try in order
    fence_starts = [
        "```python\n",
        "```py\n",
        "```Python\n",
        "```PY\n",
        "```\n",
    ]
    for start in fence_starts:
        if start in content:
            try:
                return content.split(start, 1)[1].split("```", 1)[0]
            except Exception:
                pass

    # If there are any triple backticks at all, attempt generic extraction
    if "```" in content:
        parts = content.split("```")
        # take the first non-empty middle segment
        for segment in parts[1:]:
            if segment.strip():
                return segment

    # As a last resort, if content looks like code, return it as-is
    likely_code_markers = ("def ", "class ", "import ", "from ")
    if any(marker in content for marker in likely_code_markers):
        return content
    return None

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
                stream=False,
            )
            # For stream=False, response is a single object
            
            if hasattr(response, "content") and response.content:
                content = response.content
            else:
                content = str(response)
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
    # Treat non-1 as failure
    if isSuccess != 1:
        print(f"Skipping file {file_path} due to generation error or token limit.")
        with open(UNPROCESSED_FILES, "a", encoding="utf-8") as f:
            f.write(file_path + "\n")
        return
    content = modified_code.content if hasattr(modified_code, 'content') else modified_code
    # If content is a list of TextBlock, join their .text
    if isinstance(content, list):
        content = "".join(getattr(block, "text", str(block)) for block in content)
    code_block = extract_code_from_content(content)
    if code_block is None:
        print(f"Skipping file {file_path} due to unexpected format")
        unexpected_path = "Files_not_for_root_directories/unexpected_format_claude3_sonnet.txt"
        ensure_parent_directory_exists(unexpected_path)
        with open(unexpected_path, "a", encoding="utf-8") as f:
            f.write(f"File: {file_path}\nResponse:\n{content}\n{'='*40}\n")
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
    ensure_parent_directory_exists(PROCESSED_FILES_LOG)
    with open(PROCESSED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def process_files_from_directory():
    processed_files = load_processed_files()
    all_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                all_files.append(file_path)

    files_to_process = [f for f in all_files if f not in processed_files]
    total_to_process = len(files_to_process)
    processed_count = 0
    left_count = total_to_process

    for file_path in files_to_process:
        relative_dir = os.path.relpath(os.path.dirname(file_path), INPUT_DIR)
        grouped_id = relative_dir if relative_dir != "." else "root"
        process_file(file_path, grouped_id)
        processed_count += 1
        left_count -= 1
        print(f"Processed: {processed_count}, Left: {left_count}")

        time.sleep(5)
if __name__ == "__main__":
    process_files_from_directory() 