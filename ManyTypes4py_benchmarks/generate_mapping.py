import ast
import os
import json
from pathlib import Path

def extract_function_names(file_path):
    """Extract all function names from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Try to fix f-string issues before parsing
        def fix_f_strings(code):
            """Try to fix common f-string parsing issues"""
            lines = code.split('\n')
            fixed_lines = []
            for i, line in enumerate(lines):
                # Handle problematic f-strings with unmatched parentheses
                if ('f"' in line or "f'" in line) and line.count('(') != line.count(')'):
                    # Replace problematic f-strings with regular strings
                    line = line.replace('f"', '"').replace("f'", "'")
                    print(f"    - Fixed f-string on line {i+1}")
                fixed_lines.append(line)
            return '\n'.join(fixed_lines)
        
        # Try parsing with different approaches
        tree = None
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Try with fixed f-strings
            try:
                fixed_source = fix_f_strings(source_code)
                tree = ast.parse(fixed_source)
            except SyntaxError:
                # Try with aggressive f-string removal
                try:
                    aggressive_fixed = source_code.replace('f"', '"').replace("f'", "'")
                    tree = ast.parse(aggressive_fixed)
                except SyntaxError:
                    print(f"  - Could not parse {file_path.name} due to syntax errors")
                    return []
        
        function_names = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip built-in and special methods
                if not node.name.startswith('__') and node.name not in {
                    'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                    'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
                    'abs', 'max', 'min', 'sum', 'all', 'any', 'isinstance', 'hasattr',
                    'getattr', 'setattr', 'delattr', 'super', 'property', 'staticmethod',
                    'classmethod', 'open', 'input', 'eval', 'exec', 'compile', 'hash',
                    'id', 'type', 'dir', 'vars', 'locals', 'globals', 'help', 'repr'
                }:
                    function_names.append(node.name)
        
        return function_names
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def generate_mapping():
    """Generate mapping between original and renamed function names"""
    original_dir = "Hundrad_original_typed_benchmarks"
    renamed_dir = "Hundrad_renamed_benchmarks"
    
    if not os.path.exists(original_dir):
        print(f"Directory {original_dir} does not exist.")
        return
    
    if not os.path.exists(renamed_dir):
        print(f"Directory {renamed_dir} does not exist.")
        return
    
    # Get all Python files
    original_files = list(Path(original_dir).glob("*.py"))
    renamed_files = list(Path(renamed_dir).glob("*.py"))
    
    print(f"Found {len(original_files)} original files and {len(renamed_files)} renamed files")
    
    all_mappings = {}
    
    for original_file in original_files:
        renamed_file = Path(renamed_dir) / original_file.name
        
        if not renamed_file.exists():
            print(f"Warning: No renamed version found for {original_file.name}")
            continue
        
        print(f"Processing: {original_file.name}")
        
        # Extract function names from both files
        original_functions = extract_function_names(original_file)
        renamed_functions = extract_function_names(renamed_file)
        
        print(f"  Original functions: {len(original_functions)}")
        print(f"  Renamed functions: {len(renamed_functions)}")
        
        # Create mapping for this file
        file_mapping = {}
        
        print(f"  Original functions: {original_functions}")
        print(f"  Renamed functions: {renamed_functions}")
        
        # Try different matching strategies
        if len(original_functions) == len(renamed_functions):
            # Strategy 1: Position-based matching
            for i, original_name in enumerate(original_functions):
                renamed_name = renamed_functions[i]
                if renamed_name.startswith('func_') and len(renamed_name) == 13:
                    file_mapping[original_name] = renamed_name
                    print(f"    {original_name} -> {renamed_name}")
        else:
            # Strategy 2: Try to match by function signature similarity
            # For now, just map the first few functions if they match the pattern
            min_count = min(len(original_functions), len(renamed_functions))
            for i in range(min_count):
                original_name = original_functions[i]
                renamed_name = renamed_functions[i]
                if renamed_name.startswith('func_') and len(renamed_name) == 13:
                    file_mapping[original_name] = renamed_name
                    print(f"    {original_name} -> {renamed_name}")
        
        if file_mapping:
            all_mappings[original_file.name] = file_mapping
        else:
            print(f"  No mappings found for {original_file.name}")
    
    # Save mappings to JSON file
    mapping_file = os.path.join(renamed_dir, "function_mappings.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(all_mappings, f, indent=2)
    
    print(f"\nMapping saved to: {mapping_file}")
    print(f"Total files with mappings: {len(all_mappings)}")
    
    # Print summary
    total_mappings = sum(len(mapping) for mapping in all_mappings.values())
    print(f"Total function mappings: {total_mappings}")

if __name__ == "__main__":
    generate_mapping()
