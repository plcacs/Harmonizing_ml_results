#!/usr/bin/env python3
"""
Script to analyze type patterns in JSON files containing type information.
Analyzes:
1. Parameters typed as any/Any
2. Parameters typed as List[Any]/Dict[Any] or similar formats
3. Parameters typed as Optional
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any as AnyType

def load_json_file(file_path: str) -> Dict:
    """Load JSON file and return the data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def analyze_type_patterns(data: Dict) -> Dict[str, int]:
    """Analyze type patterns in the data."""
    results = {
        'total_parameters': 0,
        'typed_parameters': 0,  # Parameters that have actual type annotations
        'any_types': 0,
        'list_dict_any': 0,
        'optional_types': 0,
        'examples': {
            'any_types': [],
            'list_dict_any': [],
            'optional_types': []
        }
    }
    
    for file_name, file_data in data.items():
        if not file_data:  # Skip empty files
            continue
            
        for function_name, function_data in file_data.items():
            if not isinstance(function_data, list):
                continue
                
            for param in function_data:
                if param.get('category') != 'arg':
                    continue
                    
                results['total_parameters'] += 1
                param_types = param.get('type', [])
                param_name = param.get('name', 'unknown')
                
                # Check if parameter has actual type annotations (not empty or just whitespace)
                has_type_annotation = False
                for type_str in param_types:
                    type_str = str(type_str).strip()
                    if type_str and type_str != "" and type_str != "None":
                        has_type_annotation = True
                        break
                
                if has_type_annotation:
                    results['typed_parameters'] += 1
                
                for type_str in param_types:
                    type_str = str(type_str).strip()
                    
                    # Check for any/Any types
                    if re.search(r'\bany\b', type_str, re.IGNORECASE):
                        results['any_types'] += 1
                        results['examples']['any_types'].append({
                            'file': file_name,
                            'function': function_name,
                            'param': param_name,
                            'type': type_str
                        })
                    
                    # Check for List[Any]/Dict[Any] patterns
                    if re.search(r'\b(?:List|Dict|list|dict)\s*\[.*\bany\b.*\]', type_str, re.IGNORECASE):
                        results['list_dict_any'] += 1
                        results['examples']['list_dict_any'].append({
                            'file': file_name,
                            'function': function_name,
                            'param': param_name,
                            'type': type_str
                        })
                    
                    # Check for Optional types
                    if re.search(r'\boptional\b', type_str, re.IGNORECASE) or 'Union[' in type_str:
                        results['optional_types'] += 1
                        results['examples']['optional_types'].append({
                            'file': file_name,
                            'function': function_name,
                            'param': param_name,
                            'type': type_str
                        })
    
    return results

def print_analysis_results(results: Dict, model_name: str):
    """Print analysis results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    total_params = results['total_parameters']
    if total_params == 0:
        print("No parameters found in the data.")
        return
    
    typed_params = results['typed_parameters']
    print(f"Total Parameters Analyzed: {total_params:,}")
    print(f"Parameters with Type Annotations: {typed_params:,} ({typed_params/total_params*100:.2f}% of total)")
    print(f"{'='*60}")
    
    # Any types analysis
    any_count = results['any_types']
    any_percentage = (any_count / typed_params) * 100 if typed_params > 0 else 0
    print(f"Parameters typed as any/Any: {any_count:,} ({any_percentage:.2f}% of typed params)")
    
    # List/Dict with Any analysis
    list_dict_count = results['list_dict_any']
    list_dict_percentage = (list_dict_count / typed_params) * 100 if typed_params > 0 else 0
    print(f"Parameters typed as List[Any]/Dict[Any]: {list_dict_count:,} ({list_dict_percentage:.2f}% of typed params)")
    
    # Optional types analysis
    optional_count = results['optional_types']
    optional_percentage = (optional_count / typed_params) * 100 if typed_params > 0 else 0
    print(f"Parameters typed as Optional: {optional_count:,} ({optional_percentage:.2f}% of typed params)")
    
    # Show some examples
    """print(f"\n{'='*60}")
    print("EXAMPLES:")
    print(f"{'='*60}")
    
    for pattern_type, examples in results['examples'].items():
        if examples:
            print(f"\n{pattern_type.upper().replace('_', ' ')} Examples:")
            for i, example in enumerate(examples[:5], 1):  # Show first 5 examples
                print(f"  {i}. {example['file']} -> {example['function']} -> {example['param']}: {example['type']}")
            if len(examples) > 5:
                print(f"  ... and {len(examples) - 5} more examples")"""

def main():
    """Main function to analyze all JSON files."""
    base_dir = Path(".")  # Current directory
    
    # List of JSON files to analyze
    json_files = [
        "Type_info_deep_seek_benchmarks.json",
        "Type_info_deep_seek_2nd_run_benchmarks.json", 
        "Type_info_gpt4o_benchmarks.json",
        "Type_info_gpt4o_2nd_run_benchmarks.json",
        "Type_info_o1_mini_benchmarks.json",
        "Type_info_o1_mini_2nd_run_benchmarks.json",
        "Type_info_original_files.json"
    ]
    
    print("TYPE PATTERN ANALYSIS SCRIPT")
    print("="*60)
    print("Analyzing type patterns in JSON files...")
    
    all_results = {}
    
    for json_file in json_files:
        file_path = base_dir / json_file
        if not file_path.exists():
            print(f"Warning: {json_file} not found, skipping...")
            continue
            
        print(f"\nProcessing {json_file}...")
        data = load_json_file(str(file_path))
        
        if data:
            # Extract model name from filename
            model_name = json_file.replace("Type_info_", "").replace("_benchmarks.json", "")
            results = analyze_type_patterns(data)
            all_results[model_name] = results
            print_analysis_results(results, model_name)
        else:
            print(f"Failed to load data from {json_file}")
    
    # Summary comparison
    print(f"\n{'='*120}")
    print("SUMMARY COMPARISON ACROSS ALL MODELS")
    print(f"{'='*120}")
    
    if all_results:
        print(f"{'Model':<20} {'Total Params':<15} {'Typed Params':<15} {'Any Types':<12} {'Any %':<8} {'List/Dict Any':<15} {'List/Dict %':<12} {'Optional':<12} {'Optional %':<12}")
        print("-" * 120)
        
        for model_name, results in all_results.items():
            total = results['total_parameters']
            typed_params = results['typed_parameters']
            any_count = results['any_types']
            list_dict_count = results['list_dict_any']
            optional_count = results['optional_types']
            
            any_percentage = (any_count / typed_params) * 100 if typed_params > 0 else 0
            list_dict_percentage = (list_dict_count / typed_params) * 100 if typed_params > 0 else 0
            optional_percentage = (optional_count / typed_params) * 100 if typed_params > 0 else 0
            
            print(f"{model_name:<20} {total:<15,} {typed_params:<15,} {any_count:<12} {any_percentage:<8.2f} {list_dict_count:<15} {list_dict_percentage:<12.2f} {optional_count:<12} {optional_percentage:<12.2f}")
    
    print(f"\n{'='*120}")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 