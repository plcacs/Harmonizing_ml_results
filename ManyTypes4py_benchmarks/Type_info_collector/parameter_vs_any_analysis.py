import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def load_json_file(file_path):
    """Load JSON file and handle potential errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_any_counts(type_info_data):
    """Extract 'Any' type counts per file from type info data."""
    any_counts = defaultdict(int)
    
    # Navigate through the JSON structure to find type information
    if isinstance(type_info_data, dict):
        for filename, functions in type_info_data.items():
            if isinstance(functions, dict):
                # Each function has a list of parameters/returns
                for func_name, func_data in functions.items():
                    if isinstance(func_data, list):
                        for param in func_data:
                            if isinstance(param, dict):
                                # Check parameter types
                                param_types = param.get('type', [])
                                if isinstance(param_types, list):
                                    for type_annotation in param_types:
                                        if isinstance(type_annotation, str) and type_annotation == 'Any':
                                            any_counts[filename] += 1
    
    return any_counts

def extract_mypy_results_with_stats(mypy_data):
    """Extract compilation status, error counts, and parameter stats from mypy results."""
    results = {}
    
    if isinstance(mypy_data, dict):
        for filename, data in mypy_data.items():
            if isinstance(data, dict):
                stats = data.get('stats', {})
                results[filename] = {
                    'error_count': data.get('error_count', 0),
                    'isCompiled': data.get('isCompiled', False),
                    'total_parameters': stats.get('total_parameters', 0),
                    'parameters_with_annotations': stats.get('parameters_with_annotations', 0)
                }
    
    return results

def create_success_rates_by_count(df, count_column):
    """Calculate success rates for each unique count value."""
    # Group by the actual count values and calculate success rates
    success_rates = df.groupby(count_column)['isCompiled'].agg(['mean', 'count']).reset_index()
    success_rates.columns = [count_column, 'success_rate', 'file_count']
    
    return success_rates

def plot_comparison_line_chart(any_success_rates, param_success_rates, model_name, output_dir):
    """Create a line plot comparing success rates by Any count vs parameter count."""
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot Any count success rates
    any_counts = any_success_rates['any_count'].tolist()
    any_rates = any_success_rates['success_rate'].tolist()
    plt.plot(any_counts, any_rates, 'o-', linewidth=2, markersize=8, 
             label='Any Count', color='blue')
    
    # Plot parameter count success rates
    param_counts = param_success_rates['parameters_with_annotations'].tolist()
    param_rates = param_success_rates['success_rate'].tolist()
    plt.plot(param_counts, param_rates, 's-', linewidth=2, markersize=8, 
             label='Parameters with Annotations', color='red')
    
    # Customize the plot
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Success Rate (0-1)', fontsize=12)
    plt.title(f'Compilation Success Rates: Any Count vs Parameters with Annotations\n{model_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PDF
    safe_model_name = model_name.replace(' ', '_').replace('-', '_')
    pdf_filename = os.path.join(output_dir, f"{safe_model_name}_Any_vs_Parameters_comparison.pdf")
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {pdf_filename}")
    
    plt.show()

def main():
    # Load data files for all models
    print("Loading data files...")
    
    typed_info_files={"Calude3.7":"ManyTypes4py_benchmarks\Type_info_collector\Type_info_claude3_sonnet_1st_run_benchmarks.json",
                "Gpt4O":"ManyTypes4py_benchmarks\Type_info_collector\Type_info_gpt4o_benchmarks.json",
                "O1-mini":"ManyTypes4py_benchmarks\Type_info_collector\Type_info_o1_mini_benchmarks.json",
                "O3-mini":"ManyTypes4py_benchmarks\Type_info_collector\Type_info_o3_mini_1st_run_benchmarks.json",
                "Deepseek":"ManyTypes4py_benchmarks\Type_info_collector\Type_info_deep_seek_benchmarks.json",
                "Human":"ManyTypes4py_benchmarks\Type_info_collector\Type_info_original_files.json"
                }
    mypy_files={"Calude3.7":"ManyTypes4py_benchmarks\mypy_results\mypy_outputs\mypy_results_claude3_sonnet_1st_run_with_errors.json",
                "Gpt4O":"ManyTypes4py_benchmarks\mypy_results\mypy_outputs\mypy_results_gpt4o_with_errors.json",
                "O1-mini":"ManyTypes4py_benchmarks\mypy_results\mypy_outputs\mypy_results_o1_mini_with_errors.json",
                "O3-mini":"ManyTypes4py_benchmarks\mypy_results\mypy_outputs\mypy_results_o3_mini_1st_run_with_errors.json",
                "Deepseek":"ManyTypes4py_benchmarks\mypy_results\mypy_outputs\mypy_results_deepseek_with_errors.json",
                "Human":"ManyTypes4py_benchmarks\mypy_results\mypy_outputs\mypy_results_original_files_with_errors.json"
                }
    
    # Create output directory
    import os
    output_dir = "ManyTypes4py_benchmarks/Type_info_collector/visualizations/Any_VS_param_count_VS_succes_ratio"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model's data
    for model_name in typed_info_files.keys():
        print(f"\n{'='*50}")
        print(f"Processing {model_name}")
        print(f"{'='*50}")
        
        # Load data for this model
        type_info_data = load_json_file(typed_info_files[model_name])
        mypy_data = load_json_file(mypy_files[model_name])
        
        if not type_info_data or not mypy_data:
            print(f"Failed to load required data files for {model_name}")
            continue
        
        print("Extracting data...")
        
        # Extract mypy results with parameter stats
        mypy_results = extract_mypy_results_with_stats(mypy_data)
        print(f"Found {len(mypy_results)} files with mypy results")
        
        # Extract Any counts from type info data
        any_counts = extract_any_counts(type_info_data)
        print(f"Found {len(any_counts)} files with Any count information")
        
        # Prepare combined dataset
        data = []
        for filename in mypy_results.keys():
            if filename in any_counts:
                data.append({
                    'filename': filename,
                    'any_count': any_counts[filename],
                    'total_parameters': mypy_results[filename]['total_parameters'],
                    'parameters_with_annotations': mypy_results[filename]['parameters_with_annotations'],
                    'isCompiled': mypy_results[filename]['isCompiled'],
                    'error_count': mypy_results[filename]['error_count']
                })
        
        if not data:
            print(f"No matching data found for {model_name}")
            continue
        
        df = pd.DataFrame(data)
        print(f"Combined dataset has {len(df)} files")
        
        # Create success rates for Any count
        any_success_rates = create_success_rates_by_count(df, 'any_count')
        print("\nSuccess rates by Any count:")
        print(any_success_rates)
        
        # Create success rates for parameters with annotations
        param_success_rates = create_success_rates_by_count(df, 'parameters_with_annotations')
        print("\nSuccess rates by parameters with annotations:")
        print(param_success_rates)
        
        # Create the comparison plot and save as PDF
        print(f"\nCreating comparison plot for {model_name}...")
        plot_comparison_line_chart(any_success_rates, param_success_rates, model_name, output_dir)

if __name__ == "__main__":
    main()
