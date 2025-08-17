import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


def load_json_file(file_path):
    """Load JSON file and handle potential errors."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_complexity_data(complexity_data):
    """Extract average CCN per file from complexity analysis data."""
    complexity_info = {}
    
    if isinstance(complexity_data, dict):
        for filename, data in complexity_data.items():
            if isinstance(data, dict):
                complexity_info[filename] = {
                    "average_CCN": data.get("average_CCN", 0),
                    "total_line_count": data.get("total_line_count", 0),
                    "function_count": data.get("function_count", 0)
                }
    
    return complexity_info


def extract_mypy_results(mypy_data):
    """Extract compilation status and error counts from mypy results."""
    results = {}
    
    if isinstance(mypy_data, dict):
        for filename, data in mypy_data.items():
            if isinstance(data, dict):
                results[filename] = {
                    "isCompiled": data.get("isCompiled", False),
                    "error_count": data.get("error_count", 0)
                }
    
    return results


def plot_complexity_vs_compilation(df, model_name, output_dir):
    """Create scatter plot of average CCN vs compilation success."""
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot colored by compilation success
    scatter = plt.scatter(
        df["average_CCN"],
        df["total_line_count"],
        c=df["isCompiled"],
        cmap="RdYlGn",
        alpha=0.6,
        s=50
    )
    
    plt.xlabel("Average Cyclomatic Complexity Number (CCN)", fontsize=12)
    plt.ylabel("Total Line Count", fontsize=12)
    plt.title(f"File Complexity vs Compilation Success\n{model_name}", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Compilation Success", fontsize=10)
    
    plt.tight_layout()
    
    # Save as PDF
    safe_model_name = model_name.replace(" ", "_").replace("-", "_")
    pdf_filename = os.path.join(output_dir, f"{safe_model_name}_complexity_vs_compilation.pdf")
    #plt.savefig(pdf_filename, format="pdf", dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_complexity_distribution(df, model_name, output_dir):
    """Create histogram showing success and failure counts by complexity bins."""
    
    plt.figure(figsize=(12, 8))
    
    # Create bins for complexity
    bins = np.linspace(df["average_CCN"].min(), df["average_CCN"].max(), 21)  # 20 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate success and failure percentages for each bin
    success_percentages = []
    failure_percentages = []
    total_files_per_bin = []
    
    for i in range(len(bins) - 1):
        bin_mask = (df["average_CCN"] >= bins[i]) & (df["average_CCN"] < bins[i + 1])
        bin_data = df[bin_mask]
        
        if len(bin_data) > 0:
            success_count = bin_data["isCompiled"].sum()
            failure_count = len(bin_data) - success_count
            total_files = len(bin_data)
            
            success_percentage = (success_count / total_files) * 100
            failure_percentage = (failure_count / total_files) * 100
            
            success_percentages.append(success_percentage)
            failure_percentages.append(failure_percentage)
            total_files_per_bin.append(total_files)
        else:
            success_percentages.append(0)
            failure_percentages.append(0)
            total_files_per_bin.append(0)
    
    # Set up bar positions
    x = np.arange(len(bin_centers))
    width = 0.35
    
    # Create grouped bar plot
    success_bars = plt.bar(x - width/2, success_percentages, width, label='Compiled Successfully', color='green', alpha=0.7)
    failure_bars = plt.bar(x + width/2, failure_percentages, width, label='Compilation Failed', color='red', alpha=0.7)
    
    # Add text labels showing percentages on bars
    for i, (success_bar, failure_bar, total_files) in enumerate(zip(success_bars, failure_bars, total_files_per_bin)):
        if total_files > 0:
            # Add success percentage
            if success_bar.get_height() > 0:
                plt.text(success_bar.get_x() + success_bar.get_width()/2, success_bar.get_height() + 0.5, 
                        f'{success_bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
            # Add failure percentage
            if failure_bar.get_height() > 0:
                plt.text(failure_bar.get_x() + failure_bar.get_width()/2, failure_bar.get_height() + 0.5, 
                        f'{failure_bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
            # Add total count above the bin
            plt.text(i, max(success_bar.get_height(), failure_bar.get_height()) + 2, 
                    f'n={total_files}', ha='center', va='bottom', fontsize=8, weight='bold')
    
    plt.xlabel("Complexity Bins (Average CCN)", fontsize=12)
    plt.ylabel("Percentage of Files", fontsize=12)
    plt.title(f"Compilation Success vs Failure by Complexity Bins\n{model_name}", fontsize=14)
    plt.xticks(x, [f'{bin_centers[i]:.1f}' for i in range(len(bin_centers))], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PDF
    safe_model_name = model_name.replace(" ", "_").replace("-", "_")
    pdf_filename = os.path.join(output_dir, f"{safe_model_name}_complexity_distribution_percentage.pdf")
    plt.savefig(pdf_filename, format="pdf", dpi=300, bbox_inches="tight")
    
    #plt.show()


def save_statistics(df, model_name, output_dir):
    """Save statistical analysis to CSV file."""
    
    # Calculate statistics
    compiled_stats = df[df["isCompiled"] == True]["average_CCN"].describe()
    failed_stats = df[df["isCompiled"] == False]["average_CCN"].describe()
    
    # Create summary DataFrame
    summary_data = {
        "Metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        "Compiled_Files": compiled_stats.values,
        "Failed_Files": failed_stats.values
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    safe_model_name = model_name.replace(" ", "_").replace("-", "_")
    csv_filename = os.path.join(output_dir, f"{safe_model_name}_complexity_statistics.csv")
    summary_df.to_csv(csv_filename, index=False)
    
    # Also save the full dataset
    full_data_filename = os.path.join(output_dir, f"{safe_model_name}_full_complexity_data.csv")
    df.to_csv(full_data_filename, index=False)
    
    print(f"Statistics saved for {model_name}")
    print(f"Compiled files: {len(df[df['isCompiled'] == True])}")
    print(f"Failed files: {len(df[df['isCompiled'] == False])}")
    print(f"Average CCN - Compiled: {compiled_stats['mean']:.2f}, Failed: {failed_stats['mean']:.2f}")


def main():
    # Define file mappings
    """complexity_files = {
        "GPT4O": "ManyTypes4py_benchmarks/complexity_of_source_codes/gpt4o_complexity_analysis.json",
        "O1-mini": "ManyTypes4py_benchmarks/complexity_of_source_codes/o1_mini_complexity_analysis.json",
        "O3-mini": "ManyTypes4py_benchmarks/complexity_of_source_codes/o3_mini_1st_run_complexity_analysis.json",
        "Deepseek": "ManyTypes4py_benchmarks/complexity_of_source_codes/deep_seek_complexity_analysis.json",
        "Human": "ManyTypes4py_benchmarks/complexity_of_source_codes/original_files_complexity_analysis.json",
        "gpt35_2nd_run": "ManyTypes4py_benchmarks/complexity_of_source_codes/gpt35_2nd_run_complexity_analysis.json"
    }"""
    complexity_files = {
         "Human": "ManyTypes4py_benchmarks/complexity_of_source_codes/original_files_complexity_analysis.json",
    }
    
    mypy_files = {
        "GPT4O": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
        "O1-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
        "O3-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        "Deepseek": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
        "Human": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_original_files_with_errors.json",
        "gpt35_2nd_run": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt35_2nd_run_with_errors.json",
        "claude3_sonnet_1st_run": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json"
    
    }
    
    # Create output directory
    output_dir = "ManyTypes4py_benchmarks/complexity_of_source_codes/complexity_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load complexity data once (using Human data for all models)
    complexity_data = load_json_file(complexity_files["Human"])
    if not complexity_data:
        print("Error: Could not load complexity data")
        return
    
    complexity_info = extract_complexity_data(complexity_data)
    
    # Process each model's mypy data
    for model_name in mypy_files.keys():
        print(f"\nProcessing {model_name}...")
        
        # Load mypy data for this model
        mypy_data = load_json_file(mypy_files[model_name])
        
        if not mypy_data:
            print(f"Skipping {model_name} due to missing mypy data")
            continue
        
        # Extract mypy results
        mypy_results = extract_mypy_results(mypy_data)
        
        # Prepare combined dataset
        data = []
        for filename in mypy_results.keys():
            if filename in complexity_info:
                data.append({
                    "filename": filename,
                    "average_CCN": complexity_info[filename]["average_CCN"],
                    "total_line_count": complexity_info[filename]["total_line_count"],
                    "function_count": complexity_info[filename]["function_count"],
                    "isCompiled": mypy_results[filename]["isCompiled"],
                    "error_count": mypy_results[filename]["error_count"]
                })
        
        if not data:
            print(f"No matching data found for {model_name}")
            continue
        
        df = pd.DataFrame(data)
        
        # Create visualizations
        #plot_complexity_vs_compilation(df, model_name, output_dir)
        plot_complexity_distribution(df, model_name, output_dir)
        
        # Save statistics
        #save_statistics(df, model_name, output_dir)


if __name__ == "__main__":
    main()
