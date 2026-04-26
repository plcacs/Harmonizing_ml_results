"""
Aggregate Report Generator for Original vs Strict AST Comparison
Combines results from all 17 folders and provides summary statistics.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def generate_aggregate_report(input_dir: str = "./original_vs_strict_comparison", 
                             output_dir: str = "./original_vs_strict_comparison"):
    """Generate aggregate report from all folder comparison results."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Aggregate statistics
    total_files = 0
    total_differences = 0
    files_with_changes = []
    changes_by_type = defaultdict(int)
    
    # Per-category tracking
    class_matches = {"yes": 0, "no": 0}
    method_matches = {"yes": 0, "no": 0}
    param_matches = {"yes": 0, "no": 0}
    control_matches = {"yes": 0, "no": 0}
    
    # Folder-level stats
    folder_stats = {}
    
    print("Processing comparison results...")
    
    # Process each folder's results
    for folder_num in range(1, 18):
        csv_file = input_path / f"comparison_results_{folder_num}.csv"
        
        if not csv_file.exists():
            print(f"Warning: {csv_file.name} not found")
            continue
        
        folder_files = 0
        folder_changes = 0
        folder_diff_files = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_files += 1
                folder_files += 1
                
                # Track matches
                class_matches[row['classes_match']] += 1
                method_matches[row['methods_match']] += 1
                param_matches[row['parameters_match']] += 1
                control_matches[row['control_flow_match']] += 1
                
                # Track differences
                diff_count = int(row['total_differences']) if row['total_differences'] else 0
                total_differences += diff_count
                
                if diff_count > 0:
                    folder_changes += 1
                    files_with_changes.append({
                        'folder': folder_num,
                        'filename': row['filename'],
                        'differences': diff_count,
                        'changes': row['structural_changes_found']
                    })
                    folder_diff_files.append(row['filename'])
                    
                    # Track change types
                    if row['classes_match'] == 'no':
                        changes_by_type['class_mismatch'] += 1
                    if row['methods_match'] == 'no':
                        changes_by_type['method_mismatch'] += 1
                    if row['parameters_match'] == 'no':
                        changes_by_type['parameter_mismatch'] += 1
                    if row['control_flow_match'] == 'no':
                        changes_by_type['control_flow_mismatch'] += 1
        
        folder_stats[folder_num] = {
            'total_files': folder_files,
            'files_with_changes': folder_changes,
            'change_percentage': (folder_changes / folder_files * 100) if folder_files > 0 else 0,
            'changed_files': folder_diff_files
        }
    
    # Generate summary report
    summary_report = {
        'summary': {
            'total_files_compared': total_files,
            'total_differences_found': total_differences,
            'files_with_structural_changes': len(files_with_changes),
            'change_rate': f"{len(files_with_changes) / total_files * 100:.2f}%" if total_files > 0 else "0%"
        },
        'match_statistics': {
            'classes': class_matches,
            'methods': method_matches,
            'parameters': param_matches,
            'control_flow': control_matches
        },
        'changes_by_type': dict(changes_by_type),
        'folder_statistics': folder_stats
    }
    
    # Write JSON summary
    summary_file = output_path / "aggregate_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2)
    print(f"✓ Summary saved to {summary_file}")
    
    # Write detailed report (files with changes)
    changes_file = output_path / "files_with_changes.csv"
    with open(changes_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['folder', 'filename', 'differences', 'changes'])
        writer.writeheader()
        for item in sorted(files_with_changes, key=lambda x: (-x['differences'], x['folder'])):
            writer.writerow(item)
    print(f"✓ Changed files report saved to {changes_file} ({len(files_with_changes)} files)")
    
    # Print console summary
    print("\n" + "="*70)
    print("AGGREGATE COMPARISON REPORT - Original vs Strict (gpt5_4_run)")
    print("="*70)
    print(f"\nTotal Files Compared: {total_files}")
    print(f"Files with Structural Changes: {len(files_with_changes)} ({len(files_with_changes) / total_files * 100:.2f}%)")
    print(f"Total Differences Found: {total_differences}")
    
    print(f"\nMATCH STATISTICS:")
    print(f"  Classes:      {class_matches['yes']} match, {class_matches['no']} mismatch")
    print(f"  Methods:      {method_matches['yes']} match, {method_matches['no']} mismatch")
    print(f"  Parameters:   {param_matches['yes']} match, {param_matches['no']} mismatch")
    print(f"  Control Flow: {control_matches['yes']} match, {control_matches['no']} mismatch")
    
    print(f"\nCHANGES BY TYPE:")
    for change_type, count in sorted(changes_by_type.items(), key=lambda x: -x[1]):
        print(f"  {change_type}: {count}")
    
    print(f"\nFOLDER BREAKDOWN:")
    for folder_num in range(1, 18):
        if folder_num in folder_stats:
            stats = folder_stats[folder_num]
            print(f"  Folder {folder_num:2d}: {stats['total_files']:2d} files, {stats['files_with_changes']:2d} with changes ({stats['change_percentage']:5.1f}%)")
    
    print("\n" + "="*70)
    print(f"Reports generated in: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_aggregate_report()
    print("Aggregate report generation complete!")
