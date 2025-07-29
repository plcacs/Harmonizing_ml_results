#!/usr/bin/env python3
"""
Script to split mypy results based on parameter annotation conditions.
Takes mypy_results_original_files_with_errors.json as input and creates two separate JSON files:
1. Files where total_parameters > 0 but parameters_with_annotations == 0
2. Files where the above condition is not met
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

def split_mypy_results(input_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Split mypy results into two categories based on parameter annotation conditions.
    
    Returns:
        Dictionary with two keys:
        - 'no_annotations': Files where total_parameters > 0 but parameters_with_annotations == 0
        - 'with_annotations': Files where the above condition is not met
    """
    
    # Load the input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return {}
    
    # Initialize result containers
    no_annotations = {}
    with_annotations = {}
    
    # Process each file entry
    for filename, file_data in data.items():
        stats = file_data.get('stats', {})
        total_parameters = stats.get('total_parameters', 0)
        parameters_with_annotations = stats.get('parameters_with_annotations', 0)
        
        # Check the condition: total_parameters > 0 but parameters_with_annotations == 0
        if total_parameters > 0 and parameters_with_annotations == 0:
            no_annotations[filename] = file_data
        else:
            with_annotations[filename] = file_data
    
    return {
        'no_annotations': no_annotations,
        'with_annotations': with_annotations
    }

def save_results(results: Dict[str, Dict[str, Any]], output_dir: str = "."):
    """
    Save the split results to separate JSON files.
    
    Args:
        results: Dictionary containing the split results
        output_dir: Directory to save the output files
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save files with no annotations
    no_annotations_file = os.path.join(output_dir, "files_with_no_parameter_annotations.json")
    try:
        with open(no_annotations_file, 'w', encoding='utf-8') as f:
            json.dump(results['no_annotations'], f, indent=2)
        print(f"Saved {len(results['no_annotations'])} files with no parameter annotations to: {no_annotations_file}")
    except Exception as e:
        print(f"Error saving no_annotations file: {e}")
    
    # Save files with annotations
    with_annotations_file = os.path.join(output_dir, "files_with_parameter_annotations.json")
    try:
        with open(with_annotations_file, 'w', encoding='utf-8') as f:
            json.dump(results['with_annotations'], f, indent=2)
        print(f"Saved {len(results['with_annotations'])} files with parameter annotations to: {with_annotations_file}")
    except Exception as e:
        print(f"Error saving with_annotations file: {e}")

def print_summary(results: Dict[str, Dict[str, Any]]):
    """
    Print a summary of the split results.
    
    Args:
        results: Dictionary containing the split results
    """
    
    no_annotations_count = len(results['no_annotations'])
    with_annotations_count = len(results['with_annotations'])
    total_count = no_annotations_count + with_annotations_count
    
    print(f"\n{'='*60}")
    print("SPLIT RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total_count}")
    print(f"Files with NO parameter annotations: {no_annotations_count} ({no_annotations_count/total_count*100:.2f}%)")
    print(f"Files WITH parameter annotations: {with_annotations_count} ({with_annotations_count/total_count*100:.2f}%)")
    
    # Show some examples from each category
    """
    if results['no_annotations']:
        print(f"\nSample files with NO parameter annotations:")
        for i, filename in enumerate(list(results['no_annotations'].keys())[:5], 1):
            stats = results['no_annotations'][filename]['stats']
            print(f"  {i}. {filename} (total_params: {stats['total_parameters']}, annotated: {stats['parameters_with_annotations']})")
        if len(results['no_annotations']) > 5:
            print(f"  ... and {len(results['no_annotations']) - 5} more")
    
    if results['with_annotations']:
        print(f"\nSample files WITH parameter annotations:")
        for i, filename in enumerate(list(results['with_annotations'].keys())[:5], 1):
            stats = results['with_annotations'][filename]['stats']
            print(f"  {i}. {filename} (total_params: {stats['total_parameters']}, annotated: {stats['parameters_with_annotations']})")
        if len(results['with_annotations']) > 5:
            print(f"  ... and {len(results['with_annotations']) - 5} more")"""

def main():
    """
    Main function to execute the script.
    """
    
    # Input file path
    input_file = "mypy_outputs/mypy_results_original_files_with_errors.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    print("PARAMETER ANNOTATION SPLIT SCRIPT")
    print("="*60)
    print(f"Processing input file: {input_file}")
    
    # Split the results
    results = split_mypy_results(input_file)
    
    if not results:
        print("No results to process.")
        return
    
    # Print summary
    print_summary(results)
    
    # Save results to separate files
    save_results(results, output_dir="split_original_files")
    
    print(f"\n{'='*60}")
    print("Script execution completed successfully!")

if __name__ == "__main__":
    main() 