import json
import os

# Read the original grouped_file_paths.json
with open('grouped_file_paths.json', 'r') as f:
    original_data = json.load(f)

# Create new data with updated paths
new_data = {}

for group_num, file_list in original_data.items():
    new_file_list = []
    for file_path in file_list:
        # Replace untyped_benchmarks with ManyTypes4py_benchmarks/original_files
        # and convert backslashes to forward slashes
        new_path = file_path.replace('untyped_benchmarks\\', 'ManyTypes4py_benchmarks/original_files/')
        new_path = new_path.replace('\\', '/')
        new_file_list.append(new_path)
    
    new_data[group_num] = new_file_list

# Write the new JSON file
with open('original_files_grouped_paths.json', 'w') as f:
    json.dump(new_data, f, indent=2)

print("Successfully created original_files_grouped_paths.json with all 18 groups!")
print(f"Total groups: {len(new_data)}")
print(f"Total files: {sum(len(files) for files in new_data.values())}")
