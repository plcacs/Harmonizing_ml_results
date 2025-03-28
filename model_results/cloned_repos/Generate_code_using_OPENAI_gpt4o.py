import os
import json
import time
import tiktoken
from openai import OpenAI
import hashlib

client = OpenAI()
PROCESSED_FILES_LOG = "processed_files_gpt4.json"
JSON_FILE = "filtered_python_files.json"
OUTPUT_DIR = "GPT4o_new1"
MAX_FILES = 500
max_tokens=8000
def get_token_count(text: str, model: str = "gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            return json.load(f)
    return {}

def save_processed_file(file_path, status):
    data = load_processed_files()
    data[file_path] = status
    with open(PROCESSED_FILES_LOG, "w") as f:
        json.dump(data, f, indent=2)

def generate_type_annotated_code(code: str) -> str:
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the annotated Python code. No Explanation needed."
    token_count = get_token_count(prompt, model="gpt-4") + 1
    if token_count>=max_tokens:
        print("Skipping file due to token count")
        return None
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
            return response.choices[0].message

        except Exception as e:
            error_msg = str(e)
            print(f"Error code: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return None
            elif "rate_limit_exceeded" in error_msg:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(610)
                wait_time += 30
            else:
                print(f"Error generating type-annotated code: {e}")
                return None


    print("Max retries reached. Skipping file.")
    return None

def process_file(file_path):
    processed_files = load_processed_files()
    if file_path in processed_files:
        print(f"Skipping {file_path}, already processed.")
        return False
    print(f"Processing file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            code = file.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"Skipping {file_path} due to read error: {e}")
        save_processed_file(file_path, False)
        return False

    modified_code = generate_type_annotated_code(code)
    content = modified_code.content if hasattr(modified_code, 'content') else modified_code
    if content is None:
        save_processed_file(file_path, False)
        return False
    try:
        code_block = content.split('```python\n')[1].split('```')[0]
    except IndexError:
        print(f"Skipping file {file_path} due to unexpected format")
        return False
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    filename = os.path.basename(file_path)
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    new_file_name = f"{filename}_gpt4_{file_hash}.py"
    new_file_path = os.path.join(OUTPUT_DIR, new_file_name)

    try:
        with open(new_file_path, 'w', encoding='utf-8', errors='ignore') as file:
            file.write(code_block)
    except (UnicodeEncodeError, IOError) as e:
        print(f"Skipping {file_path} due to write error: {e}")
        save_processed_file(file_path, False)
        return False

    save_processed_file(file_path, True)
    print(f"Successfully processed: {file_path}")
    return True

def process_files_from_json():
    processed_files = load_processed_files()

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            file_groups = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return

    ordered_categories = ["50+", "30-50", "20-30", "10-20", "05-10"]
    processed_count = 0

    for category in ordered_categories:
        if category in file_groups:
            for file_path in file_groups[category]:
                if processed_count >= MAX_FILES:
                    print(f"Reached processing limit of {MAX_FILES} files. Stopping.")
                    return
                if file_path in processed_files:
                    print(f"Skipping already processed file: {file_path}")
                    continue

                isTrue=process_file(file_path)
                if isTrue:
                    processed_count += 1
                time.sleep(30)

if __name__ == "__main__":
    process_files_from_json()
