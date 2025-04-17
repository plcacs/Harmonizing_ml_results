import os
import json
import time
import tiktoken
from openai import OpenAI
import hashlib

client = OpenAI(api_key="sk-4088954d38254ee4b072d590b59b0e27", base_url="https://api.deepseek.com")
PROCESSED_FILES_LOG = "processed_files_deep_seek.txt"
JSON_FILE = "analysis.json"
OUTPUT_DIR = "deep_seek"
TIMING_LOG = "deepseek_model_timings.json"


def get_token_count(text: str, model: str = "deepseek-reasoner"):
    encoding = tiktoken.get_encoding("cl100k_base")  # Use a known encoding
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

def generate_type_annotated_code(code: str) -> str:
    """Generate type annotations using DeepSeek API and log time taken."""
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the type annotated Python code. No Explanation. Your output should be directly executable by python compiler."
    
    token_count = get_token_count(prompt) + 1  # Ensure token limit safety
    max_retries = 5
    wait_time = 60
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()  # Start timing
            response = client.chat.completions.create(
                model="deepseek-chat",
                
                messages=[{"role": "system", "content": "You are a genius python programmer"},{"role": "user", "content": prompt}],
                
                stream=False
            )
            end_time = time.time()  # End timing
            
            duration = end_time - start_time
            log_timing("deepseek_annotation", duration)  # Log timing
            content = ""

            content = response.choices[0].message.content
            """reasoning_content=""
            for chunk in response:
                if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content  # Ensure `content` is not `None`
            """
                    
            return content if content else code  # Ret
        
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            if "rate_limit_exceeded" in error_msg:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                wait_time += 30
            else:
                print(f"Error generating type-annotated code: {e}")
                return code 
    
    print("Max retries reached. Skipping request.")
    return code

def process_file(file_path):
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
    modified_code = generate_type_annotated_code(code)
    end_time = time.time()
    log_timing(file_path, end_time - start_time)
    
    content = modified_code.content if hasattr(modified_code, 'content') else modified_code
    try:
        code_block = content.split('```python\n')[1].split('```')[0]
        #code_block=content
    except IndexError:
        print(f"Skipping file {file_path} due to unexpected format")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    filename = os.path.basename(file_path)
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]  # Take first 8 chars of hash
    new_file_name = f"{filename}_deepseek_{file_hash}.py"
    new_file_path = os.path.join(OUTPUT_DIR, new_file_name)
    
    try:
        with open(new_file_path, 'w', encoding='utf-8', errors='ignore') as file:
            file.write(code_block)
    except (UnicodeEncodeError, IOError) as e:
        print(f"Skipping {file_path} due to write error: {e}")
        return
    
    print(f"Successfully processed: {file_path}")
    with open(PROCESSED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()

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
