import os
import json
import time
import tiktoken
from openai import OpenAI
import hashlib

client = OpenAI()
PROCESSED_FILES_LOG = "processed_files_gpt.txt"
JSON_FILE = "analysis.json"
OUTPUT_DIR = "GPT4o"


def get_token_count(text: str, model: str = "gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def load_processed_files():
    """Load previously processed files from log."""
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_processed_file(file_path):
    """Append successfully processed file to the log."""
    with open(PROCESSED_FILES_LOG, "a") as f:
        f.write(file_path + "\n")

def generate_type_annotated_code(code: str) -> str:
    """Generate type annotations using GPT-4 with retries."""
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the annotated Python code. No Explanation needed."
    token_count = get_token_count(prompt, model="gpt-4") + 1
    
    
    max_retries = 3
    wait_time = 60

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a python programming expert."},
                          {"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            if "rate_limit_exceeded" in error_msg:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(610)
                wait_time += 30
            else:
                print(f"Error generating type-annotated code: {e}")
                return code 

    print("Max retries reached. Skipping file.")
    return code

def process_file(file_path):
    processed_files = load_processed_files()
    if file_path in processed_files:
        print(f"Skipping {file_path}, already processed.")
        return

    print(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            code = file.read()
    except Exception as e:
        print(f"Read error: {e}")
        return

    modified_code = generate_type_annotated_code(code)

    # Handle markdown or plain code
    try:
        if "```" in modified_code:
            code_block = modified_code.split("```python\n")[-1].split("```")[0]
        else:
            code_block = modified_code
    except Exception as e:
        print(f"Parsing error: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.basename(file_path)
    #file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    new_file_name = f"{filename}.py"
    new_file_path = os.path.join(OUTPUT_DIR, new_file_name)

    try:
        with open(new_file_path, 'w', encoding='utf-8', errors='ignore') as file:
            file.write(code_block)
    except Exception as e:
        print(f"Write error: {e}")
        return

    save_processed_file(file_path)
    print(f"Successfully processed: {file_path}")

def process_files_from_json():
    processed_files = load_processed_files()
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            file_map = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    processed_count = 0
    for file_path in file_map:
        
        if file_path in processed_files:
            print(f"Skipping already processed file: {file_path}")
            continue
        process_file(file_path)
        processed_count += 1
        time.sleep(30)

if __name__ == "__main__":
    process_files_from_json()
