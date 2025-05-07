import json

def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_files(model_name, base_file, model_file):
    print(f"Analyzing {model_name} with {base_file} and {model_file}")
    # Load both JSON files
    no_type = load_json_file(base_file)
    model = load_json_file(model_file)

    # Precompute sets
    all_keys = set(no_type.keys()) | set(model.keys())

    # Number of files processed by LLM (present in model)
    num_processed_by_llm = len(model)
    # Number of files not processed by LLM (present in base but not in model)
    num_not_processed_by_llm = len(set(no_type.keys()) - set(model.keys()))

    # Both failures: error_count > 0 in both
    both_failures = sum(1 for k in all_keys if k in no_type and k in model and no_type[k]['error_count'] > 0 and model[k]['error_count'] > 0)
    # Both success: error_count == 0 in both
    both_success = sum(1 for k in all_keys if k in no_type and k in model and no_type[k]['error_count'] == 0 and model[k]['error_count'] == 0)
    # LLM-only failures: error_count == 0 in base but > 0 in model
    llm_only_failures = sum(1 for k in all_keys if k in no_type and k in model and no_type[k]['error_count'] == 0 and model[k]['error_count'] > 0)
    # % LLM-Only Success
    llm_only_success_pct = 100 * (1 - (llm_only_failures / (llm_only_failures + both_success))) if (llm_only_failures + both_success) > 0 else 0

    # Print table header
    print(f"\nResults for {model_name}:")
    print("| Number of file processed by llm | Number of file could not processed by llm | both-failures | both success | llm-only failures | % LLM-Only Success |")
    print("|---|---|---|---|---|---|")
    print(f"| {num_processed_by_llm} | {num_not_processed_by_llm} | {both_failures} | {both_success} | {llm_only_failures} | {llm_only_success_pct:.2f} |\n")

if __name__ == "__main__":
    analyze_files("gpt4o", "mypy_results_no_type.json", "mypy_results_gpt4o.json") 
    analyze_files("deepseek", "mypy_results_no_type.json", "mypy_results_deepseek.json")