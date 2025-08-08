import json
import pandas as pd
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
                                        if isinstance(type_annotation, str) and 'Any' in type_annotation:
                                            any_counts[filename] += 1
    
    return any_counts

def extract_mypy_results(mypy_data):
    """Extract compilation status and error counts from mypy results."""
    results = {}
    
    if isinstance(mypy_data, dict):
        for filename, data in mypy_data.items():
            if isinstance(data, dict):
                results[filename] = {
                    'error_count': data.get('error_count', 0),
                    'isCompiled': data.get('isCompiled', False)
                }
    
    return results

def test_hypothesis_1(any_counts, mypy_results, model_name):
    """Test Hypothesis 1: More 'Any' types → More 'isCompiled': true"""
    print("=== Testing Hypothesis 1: More 'Any' types → More 'isCompiled': true ===")
    
    # Prepare data for analysis
    data = []
    for filename, any_count in any_counts.items():
        if filename in mypy_results:
            data.append({
                'filename': filename,
                'any_count': any_count,
                'isCompiled': mypy_results[filename]['isCompiled']
            })
    
    if not data:
        print("No matching data found between type info and mypy results")
        return None
    
    df = pd.DataFrame(data)
    df['model'] = model_name  # Add model name for CSV export
    
    # Calculate compilation success rate by Any count groups (more granular)
    df['any_group'] = pd.cut(df['any_count'], bins=[0, 1, 5, 10, 15, 20, float('inf')], 
                             labels=['0', '1-5', '5-10', '10-15', '15-20', '20+'])
    
    success_rates = df.groupby('any_group')['isCompiled'].agg(['mean', 'count'])
    print("\nCompilation Success Rates by Any Count Groups:")
    print(success_rates)
    
    # Correlation analysis
    correlation = df['any_count'].corr(df['isCompiled'])
    print(f"\nCorrelation between Any count and compilation success: {correlation:.4f}")
    
    return df

def test_hypothesis_2(any_counts, mypy_results):
    """Test Hypothesis 2: More 'Any' types → Less 'error_count'"""
    print("\n=== Testing Hypothesis 2: More 'Any' types → Less 'error_count' ===")
    
    # Prepare data for analysis
    data = []
    for filename, any_count in any_counts.items():
        if filename in mypy_results:
            data.append({
                'filename': filename,
                'any_count': any_count,
                'error_count': mypy_results[filename]['error_count']
            })
    
    if not data:
        print("No matching data found between type info and mypy results")
        return
    
    df = pd.DataFrame(data)
    
    # Correlation analysis
    correlation = df['any_count'].corr(df['error_count'])
    print(f"Correlation between Any count and error count: {correlation:.4f}")
    
    # Group analysis
    df['any_group'] = pd.cut(df['any_count'], bins=[0, 1, 5, 10, float('inf')], 
                             labels=['0', '1-5', '6-10', '10+'])
    
    error_by_group = df.groupby('any_group')['error_count'].agg(['mean', 'std', 'count'])
    print("\nError Counts by Any Count Groups:")
    print(error_by_group)
    
    return df

def main():
    # Load data files
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
    
    # Store aggregated results for CSV export
    aggregated_results = []
    
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
        
        # Extract Any counts and mypy results
        print("Extracting Any type counts...")
        any_counts = extract_any_counts(type_info_data)
        print(f"Found {len(any_counts)} files with Any type information")
        
        print("Extracting mypy results...")
        mypy_results = extract_mypy_results(mypy_data)
        print(f"Found {len(mypy_results)} files with mypy results")
        
        # Test hypothesis 1 only
        df1 = test_hypothesis_1(any_counts, mypy_results, model_name)
        
        # Add aggregated results for CSV export
        if df1 is not None:
            # Calculate success rates by any_group for this model
            success_rates = df1.groupby('any_group')['isCompiled'].agg(['mean', 'count']).reset_index()
            success_rates['model'] = model_name
            aggregated_results.append(success_rates)
        
        # Summary for this model
        print(f"\n=== Summary for {model_name} ===")
        if df1 is not None:
            print(f"Files analyzed for Hypothesis 1: {len(df1)}")
    
    # Save aggregated results to CSV
    if aggregated_results:
        combined_agg_df = pd.concat(aggregated_results, ignore_index=True)
        
        # Create pivot table with success percentages
        pivot_df = combined_agg_df.pivot(index='model', columns='any_group', values='mean')
        # Convert to percentages
        pivot_df = pivot_df * 100
        
        # Reorder columns to match desired format
        column_order = ['0', '1-5', '5-10', '10-15', '15-20', '20+']
        pivot_df = pivot_df.reindex(columns=column_order)
        
        csv_filename = "hypothesis_1_success_percentages.csv"
        pivot_df.to_csv(csv_filename)
        print(f"\n{'='*50}")
        print(f"Success percentages saved to {csv_filename}")
        print(f"Models included: {', '.join(pivot_df.index)}")
        print(f"Format: LLM Name, 0, 1-5, 5-10, 10-15, 15-20, 20+")
        print(f"Values are success percentages (% isCompiled: true)")
        print(f"{'='*50}")
        
        # Also save the detailed aggregated results
        detailed_csv_filename = "hypothesis_1_aggregated_results.csv"
        combined_agg_df.to_csv(detailed_csv_filename, index=False)
        print(f"Detailed results also saved to {detailed_csv_filename}")

if __name__ == "__main__":
    main() 