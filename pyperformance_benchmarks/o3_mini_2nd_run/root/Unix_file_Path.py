import os
import json
from typing import List

OUTPUT_JSON: str = "python_files.json"

def collect_python_files() -> List[str]:
    py_files: List[str] = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                full_path: str = os.path.abspath(os.path.join(root, file)).replace('\\', '/')
                py_files.append(full_path)
    return py_files

def save_to_json(py_files: List[str]) -> None:
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(py_files, f, indent=2)

def main() -> None:
    py_files: List[str] = collect_python_files()
    save_to_json(py_files)
    print(f"Saved {len(py_files)} Python file paths to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()