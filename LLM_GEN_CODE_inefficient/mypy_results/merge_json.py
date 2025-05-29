import json
def merge_json_files(input_files, output_file):
    merged_data = {}
    
    for file in input_files:
        with open(file, 'r') as f:
            data = json.load(f)
            # Filter out entries with "_no_types.py" suffix
            filtered_data = {k: v for k, v in data.items() if not k.endswith("_no_types.py")}
            merged_data.update(filtered_data)
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
        print(f"Total number of merged instances: {len(merged_data)}")

if __name__ == "__main__":
    # Example usage
    input_files = ["mypy_results_o1_mini_1.json", "mypy_results_o1_mini_2.json", "mypy_results_o1_mini_3.json"]  # List your JSON files here
    output_file = "merged_mypy_results_o1_mini.json"
    merge_json_files(input_files, output_file) 
   