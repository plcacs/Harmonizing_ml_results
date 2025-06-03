import json

def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def has_syntax_error(errors):
    return any("syntax" in error.lower() for error in errors)

def analyze_files(model_name, base_file, model_file):
    print(f"Analyzing {model_name} with {base_file} and {model_file}")
    # Load both JSON files
    no_type = load_json_file(base_file)
    model = load_json_file(model_file)

    # Precompute sets
    all_keys = set(no_type.keys())

    # Get files with syntax errors
    syntax_error_files = {k for k in model if has_syntax_error(model[k].get('errors', []))}
    
    # Number of files processed by LLM (present in model, excluding syntax errors)
    num_processed_by_llm = len(no_type)
    # Number of files not processed by LLM (present in base but not in model, plus files with syntax errors)
    num_not_processed_by_llm = len(set(no_type.keys()) - set(model.keys())) + len(syntax_error_files)

    # Categorize files
    both_failures_files = [k for k in all_keys if k in no_type and k in model and no_type[k]['error_count'] > 0 and model[k]['error_count'] > 0]
    both_success_files = [k for k in all_keys if k in no_type and k in model and no_type[k]['error_count'] == 0 and model[k]['error_count'] == 0]
    llm_only_failures_files = [k for k in all_keys if k in no_type and k in model and 
                             no_type[k]['error_count'] == 0 and model[k]['error_count'] > 0 and 
                             not has_syntax_error(model[k].get('errors', []))]
    llm_only_success_files = [k for k in all_keys if k in no_type and k in model and 
                            no_type[k]['error_count'] > 0 and model[k]['error_count'] == 0 and 
                            not has_syntax_error(model[k].get('errors', []))]

    # Save results to JSON
    results = {
        "num_processed_by_llm": num_processed_by_llm,
        "num_not_processed_by_llm": num_not_processed_by_llm,
        "both_failures": len(both_failures_files),
        "both_success": len(both_success_files),
        "llm_only_failures": len(llm_only_failures_files),
        "llm_only_success": len(llm_only_success_files),
        "files": {
            "both_failures": both_failures_files,
            "both_success": both_success_files,
            "llm_only_failures": llm_only_failures_files,
            "llm_only_success": llm_only_success_files
        }
    }

    with open(f"analysis_{model_name}.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print table header
    print(f"\nResults for {model_name}:")
    print("| Number of file processed by llm | Number of file could not processed by llm | both-failures | both success | llm-only failures | % LLM-Only Success |")
    print("|---|---|---|---|---|---|")
    print(f"| {num_processed_by_llm} | {num_not_processed_by_llm} | {len(both_failures_files)} | {len(both_success_files)} | {len(llm_only_failures_files)} | {100 * (1 - (len(llm_only_failures_files) / (len(llm_only_failures_files) + len(both_success_files)))):.2f} |\n")

if __name__ == "__main__":
    analyze_files("gpt4o", "mypy_results_untyped_with_errors.json", "mypy_results_gpt4o_with_errors.json")
    analyze_files("o1-mini", "mypy_results_untyped_with_errors.json", "mypy_results_o1_mini_with_errors.json") 
    analyze_files("deepseek", "mypy_results_untyped_with_errors.json", "mypy_results_deepseek_with_errors.json")
  