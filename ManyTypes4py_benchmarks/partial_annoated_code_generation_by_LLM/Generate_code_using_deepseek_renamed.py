import os
import json
import time
import tiktoken
from openai import OpenAI
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
PROCESSED_FILES_LOG = "processed_files_deepseek_partially_typed_files.txt"
INPUT_DIR = "../partially_typed_files"
OUTPUT_DIR = "deepseek_partially_typed_files"
TIMING_LOG = "deepseek_model_timings_partially_typed_files.json"
UNPROCESSED_FILES = "unprocessed_files_deepseek_partially_typed_files.txt"

def get_token_count(text: str, model: str = "deepseek-reasoner"):
    encoding = tiktoken.get_encoding("cl100k_base")  # Use a known encoding
    return len(encoding.encode(text))

def log_timing(file_path, duration):
    """Log model processing time to a JSON file."""
    # Convert Path object to string for JSON serialization
    file_path_str = str(file_path) if hasattr(file_path, '__str__') else file_path
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

def generate_type_annotated_code(code: str) -> str:
    """Generate type annotations using DeepSeek API and log time taken."""
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the type annotated Python code. No Explanation. Your output should be directly executable by python compiler."
    
    token_count = get_token_count(prompt) + 1  # Ensure token limit safety
    max_retries = 3
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
                    
            return content if content else code  # Return content or original code
        
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

    if str(file_path) in processed_files:
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
    log_timing(str(file_path), end_time - start_time)
    
    content = modified_code.content if hasattr(modified_code, 'content') else modified_code
    try:
        code_block = content.split('```python\n')[1].split('```')[0]
    except IndexError:
        print(f"Skipping file {file_path} due to unexpected format")
        return
    
    # Create output directory structure
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    filename = file_path.name
    new_file_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        with open(new_file_path, 'w', encoding='utf-8', errors='ignore') as file:
            file.write(code_block)
    except (UnicodeEncodeError, IOError) as e:
        print(f"Skipping {file_path} due to write error: {e}")
        return
    
    print(f"Successfully processed: {file_path}")
    with open(PROCESSED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(str(file_path) + "\n")

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()

def process_all_renamed_files():
    """Process all Python files in the renamed benchmarks directory"""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # Initialize log files if they don't exist
    if not os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "w", encoding="utf-8") as f:
            f.write("")
        print(f"Created processed files log: {PROCESSED_FILES_LOG}")
    
    if not os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, "w", encoding="utf-8") as f:
            json.dump([], f)
        print(f"Created timing log: {TIMING_LOG}")
    
    if not os.path.exists(UNPROCESSED_FILES):
        with open(UNPROCESSED_FILES, "w", encoding="utf-8") as f:
            f.write("")
        print(f"Created unprocessed files log: {UNPROCESSED_FILES}")
    
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} does not exist. Please run rename_functions.py first.")
        return
    
    # Get all Python files from the renamed benchmarks directory
    python_files = list(Path(INPUT_DIR).glob("*.py"))
    
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