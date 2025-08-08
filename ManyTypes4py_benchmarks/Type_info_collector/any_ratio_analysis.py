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
        return None

def extract_any_counts(type_info_data):
    """Extract 'Any' type counts per file from type info data (only for parameters/arguments)."""
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
                                # Only count 'Any' annotations for arguments (parameters), not return types
                                category = param.get('category', '')
                                if category == 'arg':  # Only count parameter annotations
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
                    'parameters_with_annotations': stats.get('parameters_with_annotations', 0)
                }
    
    return results

def save_invalid_files(df, model_name, output_dir):
    """Save files where Any count > parameters_with_annotations to a text file."""
    invalid_files = df[df['any_count'] > df['parameters_with_annotations']]
    
    if len(invalid_files) > 0:
        txt_filename = os.path.join(output_dir, f"{model_name.replace(' ', '_').replace('-', '_')}_invalid_files.txt")
        
        with open(txt_filename, 'w') as f:
            f.write(f"Files where Any count > parameters_with_annotations for {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total invalid files: {len(invalid_files)}\n\n")
            f.write("File Name\t\tAny Count\tParameters with Annotations\tDifference\n")
            f.write("-" * 80 + "\n")
            
            for _, row in invalid_files.iterrows():
                diff = row['any_count'] - row['parameters_with_annotations']
                f.write(f"{row['filename']}\t\t{row['any_count']}\t\t{row['parameters_with_annotations']}\t\t{diff}\n")

def plot_any_vs_parameters(df, model_name, output_dir):
    """Create scatter plot of Any count vs parameters with annotations."""
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot colored by compilation success
    scatter = plt.scatter(df['any_count'], df['parameters_with_annotations'], 
                         c=df['isCompiled'], cmap='RdYlGn', alpha=0.6, s=50)
    
    plt.xlabel('Any Count', fontsize=12)
    plt.ylabel('Parameters with Annotations', fontsize=12)
    plt.title(f'Any Count vs Parameters with Annotations\n{model_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Compilation Success', fontsize=10)
    
    plt.tight_layout()
    
    # Save as PDF
    safe_model_name = model_name.replace(' ', '_').replace('-', '_')
    pdf_filename = os.path.join(output_dir, f"{safe_model_name}_Any_vs_Parameters.pdf")
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    # Load data files for all models
    typed_info_files = {
        "Claude3.7": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_claude3_sonnet_1st_run_benchmarks.json",
        #"GPT4O": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_gpt4o_benchmarks.json",
        #"O1-mini": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_o1_mini_benchmarks.json",
        #"O3-mini": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_o3_mini_1st_run_benchmarks.json",
        #"Deepseek": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_deep_seek_benchmarks.json",
        "Human": "ManyTypes4py_benchmarks/Type_info_collector/Type_info_original_files.json"
    }
    
    mypy_files = {
        "Claude3.7": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
        #"GPT4O": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_gpt4o_with_errors.json",
        #"O1-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o1_mini_with_errors.json",
        #"O3-mini": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
        #"Deepseek": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
        "Human": "ManyTypes4py_benchmarks/mypy_results/mypy_outputs/mypy_results_original_files_with_errors.json"
    }
    
    # Create output directory
    output_dir = "ManyTypes4py_benchmarks/Type_info_collector/visualizations/Any_ratio_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model's data
    for model_name in typed_info_files.keys():
        # Load data for this model
        type_info_data = load_json_file(typed_info_files[model_name])
        mypy_data = load_json_file(mypy_files[model_name])
        
        if not type_info_data or not mypy_data:
            continue
        
        # Extract mypy results with parameter stats
        mypy_results = extract_mypy_results_with_stats(mypy_data)
        
        # Extract Any counts from type info data
        any_counts = extract_any_counts(type_info_data)
        
        # Prepare combined dataset
        data = []
        for filename in mypy_results.keys():
            if filename in any_counts:
                data.append({
                    'filename': filename,
                    'any_count': any_counts[filename],
                    'parameters_with_annotations': mypy_results[filename]['parameters_with_annotations'],
                    'isCompiled': mypy_results[filename]['isCompiled'],
                    'error_count': mypy_results[filename]['error_count']
                })
        
        if not data:
            continue
        
        df = pd.DataFrame(data)
        
        # Save invalid files (where Any count > parameters_with_annotations)
        save_invalid_files(df, model_name, output_dir)
        
        # Create the scatter plot
        plot_any_vs_parameters(df, model_name, output_dir)

if __name__ == "__main__":
    main()
