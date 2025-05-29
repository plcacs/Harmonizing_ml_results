import json

def merge_json_files(input_files, output_file):
    merged_data = {}
    
    for file in input_files:
        with open(file, 'r') as f:
            data = json.load(f)
            merged_data.update(data)
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    input_files = ["file1.json", "file2.json", "file3.json"]
    output_file = "merged_output.json"
    merge_json_files(input_files, output_file) 