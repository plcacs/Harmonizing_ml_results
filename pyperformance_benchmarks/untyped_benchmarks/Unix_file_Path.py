import os
import json

OUTPUT_JSON = "python_files.json"

def collect_python_files():
    py_files = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.abspath(os.path.join(root, file)).replace('\\', '/')
                py_files.append(full_path)
    return py_files

def save_to_json(py_files):
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(py_files, f, indent=2)

def main():
    py_files = collect_python_files()
    save_to_json(py_files)
    print(f"Saved {len(py_files)} Python file paths to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
