import json
import os

print("Debug test starting...")

# Test file paths
llm_files = {
    "O3-mini": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
    "DeepSeek": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
    "Claude3-Sonnet": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",     
}

print(f"Current directory: {os.getcwd()}")

for model_name, file_path in llm_files.items():
    print(f"\nTesting {model_name}:")
    print(f"  Path: {file_path}")
    print(f"  Exists: {os.path.exists(file_path)}")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} files")
            if data:
                print(f"  Sample key: {list(data.keys())[0]}")
                sample_data = data[list(data.keys())[0]]
                print(f"  Sample data keys: {list(sample_data.keys())}")
        except Exception as e:
            print(f"  Error loading: {e}")

# Test untyped file
untyped_path = "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
print(f"\nTesting untyped file:")
print(f"  Path: {untyped_path}")
print(f"  Exists: {os.path.exists(untyped_path)}")

if os.path.exists(untyped_path):
    try:
        with open(untyped_path, 'r') as f:
            untyped_data = json.load(f)
        print(f"  Loaded {len(untyped_data)} files")
        if untyped_data:
            sample_key = list(untyped_data.keys())[0]
            print(f"  Sample key: {sample_key}")
            sample_result = untyped_data[sample_key]
            print(f"  Sample result keys: {list(sample_result.keys())}")
            print(f"  Error count: {sample_result.get('error_count')}")
            print(f"  Is compiled: {sample_result.get('isCompiled')}")
    except Exception as e:
        print(f"  Error loading: {e}")

print("Debug test complete.")



