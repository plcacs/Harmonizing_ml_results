#!/usr/bin/env python3
"""
Script to analyze drastic changes in any_ratios between different type annotation approaches.

This script compares:
- untype-input: Files with no type annotations
- partial-type-input: Files with partial type annotations (LLM generated)
- fully-type-input: Files with full type annotations (user annotated)

It identifies files where there are significant changes in any_percentage values.
"""

import json
import os
from typing import Dict, Any, List, Tuple
import argparse

def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return {}

def calculate_change_percentage(old_val: float, new_val: float) -> float:
    """Calculate percentage change between two values."""
    if old_val == 0:
        return float('inf') if new_val > 0 else 0
    return ((new_val - old_val) / old_val) * 100

def find_drastic_changes(
    untyped_data: Dict[str, Any],
    partial_data: Dict[str, Any], 
    fully_data: Dict[str, Any],
    threshold: float = 50.0
) -> List[Dict[str, Any]]:
    """
    Find files with drastic changes in any_percentage.
    
    Args:
        untyped_data: Data from untyped files
        partial_data: Data from partially typed files
        fully_data: Data from fully typed files
        threshold: Minimum percentage change to consider "drastic"
    
    Returns:
        List of dictionaries containing change information
    """
    drastic_changes = []
    
    # Get all unique file names across all datasets
    all_files = set(untyped_data.keys()) | set(partial_data.keys()) | set(fully_data.keys())
    
    for filename in all_files:
        untyped_percent = untyped_data.get(filename, {}).get('any_percentage', 0)
        partial_percent = partial_data.get(filename, {}).get('any_percentage', 0)
        fully_percent = fully_data.get(filename, {}).get('any_percentage', 0)
        
        # Calculate changes
        untyped_to_partial = calculate_change_percentage(untyped_percent, partial_percent)
        untyped_to_fully = calculate_change_percentage(untyped_percent, fully_percent)
        partial_to_fully = calculate_change_percentage(partial_percent, fully_percent)
        
        # Check if any change is drastic
        is_drastic = (
            abs(untyped_to_partial) >= threshold or
            abs(untyped_to_fully) >= threshold or
            abs(partial_to_fully) >= threshold
        )
        
        if is_drastic:
            change_info = {
                'filename': filename,
                'untyped_percent': untyped_percent,
                'partial_percent': partial_percent,
                'fully_percent': fully_percent,
                'untyped_to_partial_change': untyped_to_partial,
                'untyped_to_fully_change': untyped_to_fully,
                'partial_to_fully_change': partial_to_fully,
                'untyped_slots': untyped_data.get(filename, {}).get('any_slots', 0),
                'partial_slots': partial_data.get(filename, {}).get('any_slots', 0),
                'fully_slots': fully_data.get(filename, {}).get('any_slots', 0),
                'untyped_total': untyped_data.get(filename, {}).get('total_slots', 0),
                'partial_total': partial_data.get(filename, {}).get('total_slots', 0),
                'fully_total': fully_data.get(filename, {}).get('total_slots', 0)
            }
            drastic_changes.append(change_info)
    
    return drastic_changes

def print_summary(drastic_changes: List[Dict[str, Any]], threshold: float):
    """Print summary of drastic changes."""
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Threshold for 'drastic' change: {threshold}%")
    print(f"Total files with drastic changes: {len(drastic_changes)}")
    
    if not drastic_changes:
        print("No files found with drastic changes.")
        return
    
    # Sort by largest absolute change
    drastic_changes.sort(key=lambda x: max(
        abs(x['untyped_to_partial_change']),
        abs(x['untyped_to_fully_change']),
        abs(x['partial_to_fully_change'])
    ), reverse=True)
    
    print(f"\n=== TOP 20 MOST DRAMATIC CHANGES ===")
    for i, change in enumerate(drastic_changes[:20], 1):
        print(f"\n{i}. {change['filename']}")
        print(f"   Untyped: {change['untyped_percent']:.2f}% ({change['untyped_slots']}/{change['untyped_total']})")
        print(f"   Partial: {change['partial_percent']:.2f}% ({change['partial_slots']}/{change['partial_total']})")
        print(f"   Fully:   {change['fully_percent']:.2f}% ({change['fully_slots']}/{change['fully_total']})")
        print(f"   Changes: Untyped->Partial: {change['untyped_to_partial_change']:+.1f}%, "
              f"Untyped->Fully: {change['untyped_to_fully_change']:+.1f}%, "
              f"Partial->Fully: {change['partial_to_fully_change']:+.1f}%")

def save_detailed_results(drastic_changes: List[Dict[str, Any]], output_file: str):
    """Save detailed results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(drastic_changes, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")

def analyze_patterns(drastic_changes: List[Dict[str, Any]]):
    """Analyze patterns in the drastic changes."""
    print(f"\n=== PATTERN ANALYSIS ===")
    
    # Count different types of changes
    untyped_to_partial_increases = sum(1 for c in drastic_changes if c['untyped_to_partial_change'] > 0)
    untyped_to_partial_decreases = sum(1 for c in drastic_changes if c['untyped_to_partial_change'] < 0)
    
    untyped_to_fully_increases = sum(1 for c in drastic_changes if c['untyped_to_fully_change'] > 0)
    untyped_to_fully_decreases = sum(1 for c in drastic_changes if c['untyped_to_fully_change'] < 0)
    
    partial_to_fully_increases = sum(1 for c in drastic_changes if c['partial_to_fully_change'] > 0)
    partial_to_fully_decreases = sum(1 for c in drastic_changes if c['partial_to_fully_change'] < 0)
    
    print(f"Untyped -> Partial: {untyped_to_partial_increases} increases, {untyped_to_partial_decreases} decreases")
    print(f"Untyped -> Fully:   {untyped_to_fully_increases} increases, {untyped_to_fully_decreases} decreases")
    print(f"Partial -> Fully:   {partial_to_fully_increases} increases, {partial_to_fully_decreases} decreases")
    
    # Find files that go from 0% to high percentage
    zero_to_high = [c for c in drastic_changes 
                   if c['untyped_percent'] == 0 and (c['partial_percent'] > 20 or c['fully_percent'] > 20)]
    print(f"\nFiles going from 0% to >20%: {len(zero_to_high)}")
    
    # Find files that go from high percentage to 0%
    high_to_zero = [c for c in drastic_changes 
                   if (c['untyped_percent'] > 20 or c['partial_percent'] > 20) and c['fully_percent'] == 0]
    print(f"Files going from >20% to 0%: {len(high_to_zero)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze drastic changes in any_ratios')
    parser.add_argument('--untyped', required=True, help='Path to untyped JSON file')
    parser.add_argument('--partial', required=True, help='Path to partial typed JSON file')
    parser.add_argument('--fully', required=True, help='Path to fully typed JSON file')
    parser.add_argument('--threshold', type=float, default=50.0, help='Threshold for drastic change (default: 50%)')
    parser.add_argument('--output', default='drastic_changes_analysis.json', help='Output file for detailed results')
    
    args = parser.parse_args()
    
    print("Loading data files...")
    untyped_data = load_json_data(args.untyped)
    partial_data = load_json_data(args.partial)
    fully_data = load_json_data(args.fully)
    
    if not all([untyped_data, partial_data, fully_data]):
        print("Error: Could not load all required data files.")
        return
    
    print(f"Loaded {len(untyped_data)} untyped files, {len(partial_data)} partial files, {len(fully_data)} fully typed files")
    
    print("Analyzing drastic changes...")
    drastic_changes = find_drastic_changes(untyped_data, partial_data, fully_data, args.threshold)
    
    print_summary(drastic_changes, args.threshold)
    analyze_patterns(drastic_changes)
    save_detailed_results(drastic_changes, args.output)

if __name__ == "__main__":
    main()
