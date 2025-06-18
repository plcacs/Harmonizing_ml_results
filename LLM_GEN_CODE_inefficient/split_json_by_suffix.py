import json
import sys

def split_json_by_suffix(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    no_types = {}
    with_types = {}
    
    for filename, content in data.items():
        if filename.endswith('_no_types.py'):
            no_types[filename.replace('_no_types.py', '.py')] = content
        else:
            with_types[filename] = content
    
    # Save no_types instances
    no_types_file = input_file.replace('.json', '_no_types.json')
    with open(no_types_file, 'w') as f:
        json.dump(no_types, f, indent=4)
    
    # Save with_types instances
    with_types_file = input_file.replace('.json', '_with_types.json')
    with open(with_types_file, 'w') as f:
        json.dump(with_types, f, indent=4)
    
    print(f"Files with '_no_types.py' suffix: {len(no_types)}")
    print(f"Files without '_no_types.py' suffix: {len(with_types)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_json_by_suffix.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    split_json_by_suffix(input_file) 