import tiktoken
from openai import OpenAI
import os
import re
client = OpenAI()
import time

total_file=0
succ_count=0
def get_token_count(text: str, model: str = "gpt-4o-mini"):
    # Initialize the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Tokenize the text and return the number of tokens
    return len(encoding.encode(text))

def generate_type_annotated_code(code: str) -> str:
    global token_count,succ_count
    prompt = f"Here is a Python program:\n\n{code}\n\nAdd appropriate type annotations. Output only the updated Python code. No Explanation."
    token_count = get_token_count(prompt, model="gpt-4o-mini")
    #print(f"Input token count: {token_count}")  # Print the number of tokens used by the prompt
    token_count+=1
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" if you're using GPT-3.5
            messages=[
                {"role": "system", "content": "You are a python programming expert."},
                {"role": "user", "content": prompt},
            ],
            #max_tokens=token_count*2+50,  # max output tokens
            temperature=0,
        )
        annotated_code = response.choices[0].message
        succ_count+=1
        return annotated_code

    except Exception as e:
        print("Error generating type-annotated code: {e}")
        return code

def process_file(file_path):
    print(f"Currently processing file: {file_path}")
    with open(file_path, 'r') as file:
        code = file.read()
    modified_code = generate_type_annotated_code(code)
    
    content = modified_code.content if hasattr(modified_code, 'content') else modified_code
    
    # Split by ``` to isolate the Python code
    code_block = content.split('```python\n')[1].split('```')[0]
    if not code_block:
        code_block=code
    directory, filename = os.path.split(file_path)
    base, ext = os.path.splitext(filename)
    new_file_path = os.path.join(directory, f"{base}_gpt4o.py")
    with open(new_file_path, 'w') as file:
        file.write(code_block)
    
"""
def traverse_directories(root_dir):
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py') and not filename.endswith('_gemma.py') and not filename.endswith('OpenAIAPI.py') and not filename.endswith('_llama.py') and not filename.endswith('_gpt4o.py'):
                file_path = os.path.join(dirpath, filename)
                process_file(file_path)"""

def traverse_directories(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        # Check if any file in the current directory ends with "_gpt4o.py"
        print(dirpath,filenames)
        
        # Process files if they meet criteria
        for filename in filenames:
            if filename.endswith('.py') and not filename.endswith('_gemma.py') and not filename.endswith('OpenAIAPI.py') and not filename.endswith('_llama.py') and not filename.endswith('_gpt4o.py'):
                file_path = os.path.join(dirpath, filename)
                process_file(file_path)
if __name__ == "__main__":
    traverse_directories(os.getcwd())
    print("Total file processed: ",total_file,"Success: ",succ_count,"Failed: ",total_file-succ_count)
    