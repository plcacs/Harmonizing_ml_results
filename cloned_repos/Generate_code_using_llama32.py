import os
import json
import time
#import tiktoken
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

PROCESSED_FILES_LOG = "processed_files_llama.txt"
JSON_FILE = "filtered_python_files_linux.json"
OUTPUT_DIR = "llama_output"
TIMING_LOG = "llama_model_timings.json"
MAX_FILES = 1500

"""def get_token_count(text: str, model: str = "llama-3.2"):
    encoding = tiktoken.get_encoding("cl100k_base")  # Use a known encoding
    return len(encoding.encode(text))"""

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
    """Generate type annotations using Llama 3.2 and log time taken."""
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the type-annotated Python code. No Explanation. Your output should be directly executable by Python."
    
    #oken_count = get_token_count(prompt) + 1  # Ensure token limit safety
    max_retries = 5
    wait_time = 60
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()  # Start timing
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            input_length = inputs['input_ids'].shape[1]
            if input_length > 3000:
                print(input_length)
                print("Too Large File, skipping processing")
                return code
            outputs = model.generate(**inputs, max_length=input_length * 2 + 100, pad_token_id=tokenizer.eos_token_id)
            end_time = time.time()  # End timing
            
            duration = end_time - start_time
            log_timing("llama_annotation", duration)  # Log timing
            
            content = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return content if content else code  
        
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
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
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    filename = os.path.basename(file_path)
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]  # Take first 8 chars of hash
    new_file_name = f"{filename}_llama_{file_hash}.py"
    new_file_path = os.path.join(OUTPUT_DIR, new_file_name)
    
    try:
        with open(new_file_path, 'w', encoding='utf-8', errors='ignore') as file:
            file.write(modified_code)
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
    """Process files based on JSON file priority order."""
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
                    print("Reached processing limit. Stopping.")
                    return
                if file_path in processed_files:
                    print(f"Skipping {file_path}, already processed.")
                    continue
                process_file(file_path)
                processed_count += 1
                time.sleep(30)  # Avoid GPU overload

if __name__ == "__main__":
    process_files_from_json()
