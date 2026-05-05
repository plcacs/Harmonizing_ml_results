"""
Side-by-side Code Comparison
Shows actual code snippets from manifest_47c52e.py
"""

import ast
from pathlib import Path


def show_code_snippets():
    """Show actual code examples of the differences."""
    
    original_base = Path("./ManyTypes4py_benchmarks/original_files")
    strict_base = Path("./ManyTypes4py_benchmarks/gpt5_4_run")
    
    # Find the file
    original_files = sorted(original_base.glob("*.py"))
    files_per_folder = len(original_files) // 17
    start_idx = (6 - 1) * files_per_folder
    end_idx = start_idx + files_per_folder
    folder_originals = original_files[start_idx:end_idx]
    
    strict_file = strict_base / "6" / "manifest_47c52e.py"
    strict_folder = sorted(strict_base.glob("6/*.py"))
    file_idx = sorted([f.name for f in strict_folder]).index("manifest_47c52e.py")
    
    if file_idx < len(folder_originals):
        orig_file = folder_originals[file_idx]
        
        orig_code = orig_file.read_text(encoding='utf-8', errors='ignore')
        strict_code = strict_file.read_text(encoding='utf-8', errors='ignore')
        
        # Show first N lines
        print("\n" + "="*100)
        print("SIDE-BY-SIDE CODE COMPARISON: manifest_47c52e.py")
        print("="*100)
        
        orig_lines = orig_code.split('\n')[:50]
        strict_lines = strict_code.split('\n')[:50]
        
        print("\n[ORIGINAL FILE] (First 50 lines):")
        print("-"*100)
        for i, line in enumerate(orig_lines, 1):
            print(f"{i:3d} | {line[:96]}")
        
        print("\n" + "="*100)
        print("[STRICT VERSION] (First 50 lines):")
        print("-"*100)
        for i, line in enumerate(strict_lines, 1):
            print(f"{i:3d} | {line[:96]}")
        
        print("\n" + "="*100)
        print("FILE SIZE COMPARISON")
        print("="*100)
        print(f"Original file: {len(orig_code)} characters, {len(orig_lines)} lines")
        print(f"Strict file:   {len(strict_code)} characters, {len(strict_lines)} lines")
        print(f"Size increase: {len(strict_code) / len(orig_code) * 100:.1f}%")
        
        print("\n" + "="*100)
        print("KEY OBSERVATIONS")
        print("="*100)
        
        # Count conditionals in first 50 lines
        orig_if_count = sum(1 for line in orig_lines if 'if ' in line)
        strict_if_count = sum(1 for line in strict_lines if 'if ' in line)
        
        orig_for_count = sum(1 for line in orig_lines if 'for ' in line)
        strict_for_count = sum(1 for line in strict_lines if 'for ' in line)
        
        print(f"\nIn first 50 lines:")
        print(f"  If statements: {orig_if_count} (orig) vs {strict_if_count} (strict)")
        print(f"  For loops:     {orig_for_count} (orig) vs {strict_for_count} (strict)")
        
        print(f"\nFull file details:")
        print(f"  Original file path: {orig_file}")
        print(f"  Strict file path:   {strict_file}")


if __name__ == "__main__":
    show_code_snippets()
