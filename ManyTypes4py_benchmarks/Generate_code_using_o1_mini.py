import os
import json
import time
import tiktoken
from openai import OpenAI
import hashlib

client = OpenAI()
PROCESSED_FILES_LOG = "processed_files_o1_mini.txt"
JSON_FILE = "grouped_file_paths.json"
OUTPUT_DIR = "o1_mini"
TIMING_LOG = "o1_mini_model_timings.json"
unprocessed_files ="unprocessed_files_o1_mini.txt"
def get_token_count(text: str, model: str = "o1-mini"):
    encoding = tiktoken.encoding_for_model(model)
    #encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def log_timing(file_path, duration):
    """Log model processing time to a JSON file."""
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

def save_processed_file(file_path, status):
    data = load_processed_files()
    data[file_path] = status
    with open(PROCESSED_FILES_LOG, "w") as f:
        json.dump(data, f, indent=2)

def generate_type_annotated_code(code: str) -> str:
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the annotated Python code. No Explanation needed."
    """token_count = get_token_count(prompt, model="gpt-4o")
    if token_count > 9000:  # Add this line to skip long prompts
        print(f"Skipping file, token count too high: {token_count}")
        return code,0"""
    max_retries = 3
    wait_time = 60

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="o1-mini",
                messages=[  # No "system" message
                    {"role": "user", "content": prompt}
                ],
                # Remove temperature=0 (only default = 1 is allowed)
                # Setting it explicitly to 1 might not be necessary
            )
            return response.choices[0].message.content,1

        except Exception as e:
            error_msg = str(e)
            print(f"Error code: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return code,2
            elif "rate_limit_exceeded" in error_msg:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(610)
                wait_time += 30
            else:
                print(f"Error generating type-annotated code: {e}")
                return code,2


    print("Max retries reached. Skipping file.")
    return code,2

def process_file(file_path, grouped_id):
    """Process a single file, handling encoding errors and logging processing time."""
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
    modified_code,isSuccess = generate_type_annotated_code(code)
    end_time = time.time()
    log_timing(file_path, end_time - start_time)
    if  isSuccess==0:
        print(f"Skipping file {file_path} due to generation error or token limit.")
        with open(unprocessed_files, "a", encoding="utf-8") as f:
            f.write(file_path + "\n")
        return
    content = modified_code.content if hasattr(modified_code, 'content') else modified_code
    try:
        code_block = content.split('```python\n')[1].split('```')[0]
        #code_block=content
    except IndexError:
        print(f"Skipping file {file_path} due to unexpected format")
        return
    new_file_path= os.path.join(OUTPUT_DIR, grouped_id)
    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)
    
    filename = os.path.basename(file_path)
    #file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]  # Take first 8 chars of hash
    #new_file_name = f"{filename}_deepseek_{file_hash}.py"
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
    
    processed_count = 0
    for id_ in range(1,19):
        grouped_id=str(id_)
        for file_path in file_map[grouped_id]:
            
            if file_path in processed_files:
                print(f"Skipping already processed file: {file_path}")
                continue
            process_file(file_path, grouped_id)
            processed_count += 1
            time.sleep(300)
        time.sleep(3600)

if __name__ == "__main__":
    process_files_from_json()
