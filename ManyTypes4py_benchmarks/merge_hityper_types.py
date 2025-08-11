import json
import os
import re
from pathlib import Path
from typing import Dict, List

def parse_json_types(json_file_path: Path) -> Dict[str, Dict]:
    """Parse the JSON file and extract type information for each function."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    function_types = {}
    
    for func_name, type_entries in data.items():
        # Handle different function name formats
        if '@' in func_name:
            # Class method: "__init__@ArrowTemporalProperties" -> ("__init__", "ArrowTemporalProperties")
            if '@global' in func_name:
                clean_func_name = func_name.replace('@global', '')
            else:
                # Class method: extract just the method name
                clean_func_name = func_name.split('@')[0]
        else:
            clean_func_name = func_name
        
        func_info = {
            'args': {},
            'return_type': None
        }
        
        for entry in type_entries:
            category = entry.get('category')
            name = entry.get('name')
            types = entry.get('type', [])
            
            if category == 'arg' and name:
                # Use the raw types from JSON
                if types:
                    func_info['args'][name] = types
                else:
                    func_info['args'][name] = ['Any']
            elif category == 'return':
                if types:
                    func_info['return_type'] = types
                else:
                    func_info['return_type'] = ['Any']
        
        function_types[clean_func_name] = func_info
    
    return function_types

def convert_types_to_annotation(types: List[str]) -> str:
    """Convert a list of type strings to a simple type annotation."""
    if not types:
        return 'Any'
    
    # Clean up the type strings - just remove quotes
    cleaned_types = []
    for t in types:
        t = t.strip().strip('"\'')
        if t:
            cleaned_types.append(t)
    
    if not cleaned_types:
        return 'Any'
    
    if len(cleaned_types) == 1:
        return cleaned_types[0]
    
    # For multiple types, create a Union
    return f"Union[{', '.join(cleaned_types)}]"

def add_type_annotations(python_code: str, function_types: Dict[str, Dict]) -> str:
    """Add type annotations to Python code based on function type information."""
    lines = python_code.split('\n')
    modified_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a function definition (including class methods)
        func_match = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
        if func_match:
            indent = func_match.group(1)
            func_name = func_match.group(2)
            
            if func_name in function_types:
                func_info = function_types[func_name]
                
                # Find the end of the function signature
                signature_end = line.find(':')
                if signature_end == -1:
                    # Function signature spans multiple lines
                    signature_lines = [line]
                    j = i + 1
                    while j < len(lines) and ':' not in lines[j]:
                        signature_lines.append(lines[j])
                        j += 1
                    if j < len(lines):
                        signature_lines.append(lines[j])
                        signature_end = len(signature_lines) - 1
                        i = j
                    else:
                        # No colon found, skip this function
                        modified_lines.append(line)
                        i += 1
                        continue
                else:
                    signature_lines = [line]
                    signature_end = 0
                
                # Parse the function signature
                signature = '\n'.join(signature_lines)
                modified_signature = add_types_to_signature(signature, func_info)
                
                # Replace the signature
                if len(signature_lines) == 1:
                    modified_lines.append(modified_signature)
                else:
                    # Multi-line signature
                    modified_lines.extend(modified_signature.split('\n'))
                
                i += 1
            else:
                modified_lines.append(line)
                i += 1
        else:
            modified_lines.append(line)
            i += 1
    
    return '\n'.join(modified_lines)

def add_types_to_signature(signature: str, func_info: Dict) -> str:
    """Add type annotations to a function signature."""
    # Find the opening and closing parentheses
    open_paren = signature.find('(')
    close_paren = signature.find(')')
    
    if open_paren == -1 or close_paren == -1:
        return signature
    
    # Extract function name and parameters
    func_name_part = signature[:open_paren]
    params_part = signature[open_paren + 1:close_paren]
    rest_part = signature[close_paren:]
    
    # Parse parameters
    params = parse_parameters(params_part)
    
    # Add type annotations to parameters
    typed_params = []
    for param in params:
        param_name = param.strip()
        if '=' in param_name:
            # Parameter with default value
            name_part, default_part = param_name.split('=', 1)
            name = name_part.strip()
            default = '=' + default_part.strip()
        else:
            name = param_name.strip()
            default = ''
        
        # Add type annotation if available
        if name in func_info['args']:
            type_annotation = convert_types_to_annotation(func_info['args'][name])
            if default:
                typed_params.append(f"{name}: {type_annotation}{default}")
            else:
                typed_params.append(f"{name}: {type_annotation}")
        else:
            typed_params.append(param)
    
    # Add return type annotation
    if func_info['return_type'] and func_info['return_type'] != ['Any']:
        return_type = convert_types_to_annotation(func_info['return_type'])
        return_type_annotation = f" -> {return_type}"
    else:
        return_type_annotation = ""
    
    # Reconstruct the signature - properly handle the rest_part
    # Remove the colon from rest_part if it exists, but keep the closing parenthesis
    if rest_part.startswith(':'):
        rest_part = rest_part[1:]  # Remove the colon
    
    # Ensure we don't have duplicate closing parentheses
    if rest_part.startswith(')'):
        rest_part = rest_part[1:]  # Remove the extra closing parenthesis
    
    new_signature = f"{func_name_part}({', '.join(typed_params)}){return_type_annotation}{rest_part}"
    return new_signature

def parse_parameters(params_str: str) -> List[str]:
    """Parse function parameters from a string."""
    if not params_str.strip():
        return []
    
    params = []
    current_param = ""
    paren_count = 0
    
    for char in params_str:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == ',' and paren_count == 0:
            if current_param.strip():
                params.append(current_param.strip())
            current_param = ""
            continue
        
        current_param += char
    
    if current_param.strip():
        params.append(current_param.strip())
    
    return params

def process_file(python_file_path: Path, json_file_path: Path, output_dir: Path) -> bool:
    """Process a single Python file with its corresponding JSON file."""
    try:
        # Read the Python file
        with open(python_file_path, 'r', encoding='utf-8') as f:
            python_code = f.read()
        
        # Parse the JSON file
        function_types = parse_json_types(json_file_path)
        
        # Add type annotations
        modified_code = add_type_annotations(python_code, function_types)
        
        # Write the output file
        output_file = output_dir / python_file_path.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
        
        return True
    except Exception as e:
        print(f"Error processing {python_file_path.name}: {e}")
        return False

def main():
    """Main function to process all files."""
    # Define directories
    untyped_dir = Path("untyped_benchmarks")
    json_dir = Path("HiTyper_json_files")
    output_dir = Path("HiTyper_1st_run")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all Python files
    python_files = list(untyped_dir.glob("*.py"))
    
    print(f"Found {len(python_files)} Python files to process")
    
    processed_count = 0
    for python_file in python_files:
        # Find corresponding JSON file
        json_filename = f"{python_file.stem}_INFERREDTYPES.json"
        json_file = json_dir / json_filename
        
        if json_file.exists():
            if process_file(python_file, json_file, output_dir):
                processed_count += 1
                print(f"Processed: {python_file.name}")
        else:
            print(f"No JSON file found for: {python_file.name}")
    
    print(f"\nSuccessfully processed {processed_count} files")
    print(f"Output files saved to: {output_dir}")

if __name__ == "__main__":
    main()
