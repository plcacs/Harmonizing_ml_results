import os
import json
import time
import hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

PROCESSED_FILES_LOG = "processed_files_codellama.txt"
JSON_FILE = "untyped_files.json"
OUTPUT_DIR = "CodeLlama"

# Optional: set this in your shell instead → export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_processed_file(file_path):
    with open(PROCESSED_FILES_LOG, "a") as f:
        f.write(file_path + "\n")

def generate_type_annotated_code(code: str) -> str:
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the annotated Python code. No explanation needed."
    try:
        response = generator(prompt, max_length=1024, temperature=0.1, do_sample=False)
        return response[0]["generated_text"]
    except Exception as e:
        print(f"Error generating code: {e}")
        return code

def print_memory_summary():
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())
    else:
        print("CUDA not available.")

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

    # Print GPU memory stats
    print_memory_summary()

    # Clear CUDA cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache.")

def process_files_from_json():
    processed_files = load_processed_files()
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            file_list = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    processed_count = 0
    for file_path in file_list:
        if file_path in processed_files:
            print(f"Skipping already processed file: {file_path}")
            continue
        process_file(file_path)
        processed_count += 1
        time.sleep(10)  # increased delay to help stabilize memory

if __name__ == "__main__":
    process_files_from_json()
