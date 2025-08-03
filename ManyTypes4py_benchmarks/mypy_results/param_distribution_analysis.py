import json
import pandas as pd
import os
import glob

def analyze_param_distribution(analysis_file, mypy_file, llm_name="", analysis_type="llm_only_failures"):
    """Analyze parameter distribution for a specific LLM output"""
    
    # Load JSON files
    with open(analysis_file, "r") as f:
        analysis_data = json.load(f)

    with open(mypy_file, "r") as f:
        mypy_data = json.load(f)

    # Extract list of files based on analysis type
    if analysis_type == "llm_only_failures":
        file_list = analysis_data["files"]["llm_only_failures"]
        analysis_label = "LLM-only failures"
    elif analysis_type == "both_success":
        file_list = analysis_data["files"]["both_success"]
        analysis_label = "Both success"
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    # Collect total_parameters values
    total_params_list = []
    for file in file_list:
        if file in mypy_data and "stats" in mypy_data[file]:
            total_params_list.append(mypy_data[file]["stats"]["total_parameters"])

    # Total number of files considered
    total_count = len(total_params_list)

    # Define bins and labels
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, float('inf')]
    labels = ["1–10", "11–20", "21–30", "31–40", "41–50", "51–60",
              "61–70", "71–80", "81–100", "101–150", "151–200", "200+"]

    # Bin the data
    bin_counts = pd.cut(total_params_list, bins=bins, labels=labels, right=True).value_counts().sort_index()
    bin_percentages = (bin_counts / total_count) * 100

    # Create and display result
    result_df = pd.DataFrame({
        "Parameter Range": labels,
        "File Count": bin_counts.values,
        "Percentage": bin_percentages.values
    })
    
    if llm_name:
        print(f"\n=== {llm_name} Results ({analysis_label}) ===")
    else:
        print(f"\n=== Results for {analysis_file} ({analysis_label}) ===")
    
    print(result_df.to_string(index=False))
    print(f"Total files analyzed: {total_count}")
    
    # Additional analysis: parameters < 50 vs >= 50
    less_than_50 = sum(1 for params in total_params_list if params < 50)
    greater_equal_50 = sum(1 for params in total_params_list if params >= 50)
    
    print(f"\nParameter Count Analysis:")
    print(f"Files with < 50 parameters: {less_than_50} ({(less_than_50/total_count)*100:.1f}%)")
    print(f"Files with >= 50 parameters: {greater_equal_50} ({(greater_equal_50/total_count)*100:.1f}%)")
    
    return result_df, total_count, less_than_50, greater_equal_50

def find_llm_result_files():
    """Find all available LLM result files"""
    
    analysis_files = {"claude_3_5_sonnet": "analysis_outputs/analysis_claude_3_5_sonnet_simplified.json",
                      "o3_mini": "analysis_outputs/analysis_o3_mini_1st_run_simplified.json",
                      "o1_mini": "analysis_outputs/analysis_o1-mini_simplified.json",
                      "deepseek": "analysis_outputs/analysis_deepseek_simplified.json",
                      "gpt4o": "analysis_outputs/analysis_gpt4o_simplified.json"}
    mypy_files = {"claude_3_5_sonnet": "mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
                  "o3_mini": "mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
                  "o1_mini": "mypy_outputs/mypy_results_o1_mini_with_errors.json",
                  "deepseek": "mypy_outputs/mypy_results_deepseek_with_errors.json",
                  "gpt4o": "mypy_outputs/mypy_results_gpt4o_with_errors.json"}
    
    # Match analysis files with corresponding mypy files
    llm_results = []
    for llm_name, analysis_file in analysis_files.items():
        # Find corresponding mypy file
        corresponding_mypy = mypy_files.get(llm_name)
        
        if corresponding_mypy:
            llm_results.append((analysis_file, corresponding_mypy, llm_name))
    
    return llm_results

def main():
    """Main function to analyze multiple LLM outputs"""
    
    # Find all available LLM result files
    llm_results = find_llm_result_files()
    
    if not llm_results:
        print("No LLM result files found. Looking for files with pattern:")
        print("- *analysis*simplified.json")
        print("- *mypy_results*with_errors.json")
        return
    
    print(f"Found {len(llm_results)} LLM result sets:")
    for analysis_file, mypy_file, llm_name in llm_results:
        print(f"  - {llm_name}: {analysis_file} + {mypy_file}")
    
    # Analyze each LLM output for both llm_only_failures and both_success
    all_results = []
    
    # First analyze llm_only_failures (original analysis)
    print("=" * 60)
    print("ANALYZING LLM-ONLY FAILURES")
    print("=" * 60)
    
    for analysis_file, mypy_file, llm_name in llm_results:
        try:
            result_df, total_count, less_than_50, greater_equal_50 = analyze_param_distribution(
                analysis_file, mypy_file, llm_name, "llm_only_failures"
            )
            all_results.append((llm_name, result_df, total_count, less_than_50, greater_equal_50, "llm_only_failures"))
        except Exception as e:
            print(f"Error processing {llm_name} (llm_only_failures): {e}")
    
    # Then analyze both_success
    print("\n" + "=" * 60)
    print("ANALYZING BOTH SUCCESS")
    print("=" * 60)
    
    for analysis_file, mypy_file, llm_name in llm_results:
        try:
            result_df, total_count, less_than_50, greater_equal_50 = analyze_param_distribution(
                analysis_file, mypy_file, llm_name, "both_success"
            )
            all_results.append((llm_name, result_df, total_count, less_than_50, greater_equal_50, "both_success"))
        except Exception as e:
            print(f"Error processing {llm_name} (both_success): {e}")
    
    # Optional: Create combined summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY COMPARISON")
        print("=" * 60)
        
        # Separate summaries for each analysis type
        for analysis_type in ["llm_only_failures", "both_success"]:
            type_results = [r for r in all_results if r[5] == analysis_type]
            if type_results:
                analysis_label = "LLM-only failures" if analysis_type == "llm_only_failures" else "Both success"
                print(f"\n{analysis_label.upper()} SUMMARY:")
                print("-" * 40)
                
                summary_data = []
                for llm_name, result_df, total_count, less_than_50, greater_equal_50, _ in type_results:
                    summary_data.append({
                        "LLM": llm_name,
                        "Total Files": total_count,
                        "< 50 params": less_than_50,
                        ">= 50 params": greater_equal_50,
                        "< 50 %": f"{(less_than_50/total_count)*100:.1f}%" if total_count > 0 else "0%",
                        ">= 50 %": f"{(greater_equal_50/total_count)*100:.1f}%" if total_count > 0 else "0%"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
