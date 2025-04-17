import os
import json
import time
import tiktoken
from openai import OpenAI
import hashlib

client = OpenAI()
PROCESSED_FILES_LOG = "processed_files_o1-mini.txt"
JSON_FILE = "analysis.json"
OUTPUT_DIR = "o1-mini3"


def get_token_count(text: str, model: str = "o1-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_processed_file(file_path):
    with open(PROCESSED_FILES_LOG, "a") as f:
        f.write(file_path + "\n")

def generate_type_annotated_code(code: str) -> str:
    prompt = (
        f"Here is a Python program:\n\n{code}\n\n"
        "Add appropriate type annotations. Output only the type annotated Python code. "
        "No Explanation. Your output should be directly executable by python compiler."
    )

    token_count = get_token_count(prompt, model="o1-mini") + 1
    max_retries = 3
    wait_time = 60

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="o1-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            if "rate_limit_exceeded" in error_msg:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(610)
                wait_time += 30
            elif "model_not_found" in error_msg:
                print("Model `o1-mini` does not exist.")
                return code
            elif "unsupported_value" in error_msg:
                print("Unsupported system message/setting.")
                return code
            else:
                print(f"Error: {e}")
                return code
    print("Max retries reached.")
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
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    new_file_name = f"{filename}_o1_mini_{file_hash}.py"
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
