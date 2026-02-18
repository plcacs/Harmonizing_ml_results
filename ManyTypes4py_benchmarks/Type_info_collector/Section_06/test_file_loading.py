import json
import os

# Test file loading
test_file = "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json"

try:
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    # Write to output file
    with open('test_output.txt', 'w') as f:
        f.write(f"Successfully loaded {len(data)} files\n")
        if data:
            first_key = list(data.keys())[0]
            f.write(f"First file: {first_key}\n")
            first_data = data[first_key]
            f.write(f"Keys in first file: {list(first_data.keys())}\n")
            
            if 'functions' in first_data:
                functions = first_data['functions']
                f.write(f"Number of functions: {len(functions)}\n")
                if functions:
                    first_func = functions[0]
                    f.write(f"First function keys: {list(first_func.keys())}\n")
                    
                    if 'parameters' in first_func:
                        params = first_func['parameters']
                        f.write(f"Number of parameters: {len(params)}\n")
                        if params:
                            first_param = params[0]
                            f.write(f"First parameter: {first_param}\n")
    
    print("Test completed successfully")
    
except Exception as e:
    with open('test_error.txt', 'w') as f:
        f.write(f"Error: {str(e)}\n")
    print(f"Error: {e}")




