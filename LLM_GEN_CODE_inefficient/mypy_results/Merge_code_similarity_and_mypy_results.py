import json

def merge_json_files():
    # Read the first JSON file
    with open('merged_mypy_results_o1_mini.json', 'r') as f:
        mypy_data = json.load(f)
    
    # Read the second JSON file
    with open('o1-mini_code_similarity_old.json', 'r') as f:
        similarity_data = json.load(f)
    
    # Create merged result
    merged_result = {}
    
    # Process each file
    for filename in similarity_data.keys():
        if filename in mypy_data:
            merged_result[filename] = {
                'file_signature': sorted(similarity_data[filename]),
                'error_count': mypy_data[filename].get('error_count', 0),
                'stats': mypy_data[filename].get('stats', {})
            }
    print(f"Total number of merged instances: {len(merged_result)}")
    # Write merged result to a new file
    with open('code_similarity_o1_mini_old.json', 'w') as f:
        json.dump(merged_result, f, indent=2)

if __name__ == "__main__":
    merge_json_files() 