import json
import os


# List of analysis files to process
analysis_files = [
    "analysis_deepseek.json",
    "analysis_gpt4o.json",
    "analysis_o1-mini.json",
]

# Corresponding mypy results files
llm_mypy_results = [
    "mypy_results_deepseek_with_errors.json",
    "mypy_results_gpt4o_with_errors.json",
    "mypy_results_o1_mini_with_errors.json",
]

base_file = "mypy_results_untyped_with_errors.json"

# Create output directory if it doesn't exist
output_dir = "Filtered_type_errors"
os.makedirs(output_dir, exist_ok=True)

# Load base file data
with open(base_file) as f:
    base_data = json.load(f)

# Process each analysis file
for i, analysis_file in enumerate(analysis_files):
    print(f"\nProcessing {analysis_file}...")
    
    # Load analysis file
    with open(analysis_file) as f:
        analysis_data = json.load(f)
    
    # Load corresponding mypy results
    mypy_file = llm_mypy_results[i]
    with open(mypy_file) as f:
        mypy_data = json.load(f)
    
    # Extract files from "llm_only_failures"
    llm_only_failures = analysis_data.get("files", {}).get("llm_only_failures", [])
    print(f"Found {len(llm_only_failures)} files in llm_only_failures")
    
    # Merge results for each file
    merged_data = {}
    for fname in llm_only_failures:
        if fname in base_data and fname in mypy_data:
            merged_data[fname] = {
                "base_stats": base_data[fname]["stats"],
                "llm_stats": mypy_data[fname]["stats"],
                "base_error_count": base_data[fname]["error_count"],
                "llm_error_count": mypy_data[fname]["error_count"],
                "base_errors": base_data[fname]["errors"],
                "llm_errors": mypy_data[fname]["errors"],
            }
        else:
            print(f"Warning: File {fname} not found in either base or mypy data")
    
    # Save merged results
    
    output_file = os.path.join(
        output_dir, f"merged_{analysis_file.replace('analysis_', '').replace('.json', '')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Created merged file: {output_file} with {len(merged_data)} entries")
