print("Hello World!")
print("Testing basic functionality")

import os
print(f"Current directory: {os.getcwd()}")

# Test if we can load a simple file
test_path = "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json"
print(f"Test file exists: {os.path.exists(test_path)}")

if os.path.exists(test_path):
    import json
    with open(test_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")
    if data:
        first_key = list(data.keys())[0]
        print(f"First key: {first_key}")
        print(f"First entry keys: {list(data[first_key].keys())}")

print("Test complete!")




