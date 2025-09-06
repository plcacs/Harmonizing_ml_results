import ast
import os
import json
from pathlib import Path


def get_type_annotation(annotation):
    """Convert AST type annotation to string"""
    if annotation is None:
        return [""]

    if isinstance(annotation, ast.Name):
        return [annotation.id]
    elif isinstance(annotation, ast.Constant):
        return [str(annotation.value)]
    elif isinstance(annotation, ast.Str):  # Python < 3.8
        return [annotation.s]
    elif isinstance(annotation, ast.Subscript):
        # Handle generic types like List[str], Optional[int], etc.
        if isinstance(annotation.value, ast.Name):
            base_type = annotation.value.id
            if isinstance(annotation.slice, ast.Index):  # Python < 3.9
                slice_value = annotation.slice.value
            else:  # Python >= 3.9
                slice_value = annotation.slice

            if isinstance(slice_value, ast.Name):
                return [f"{base_type}[{slice_value.id}]"]
            elif isinstance(slice_value, ast.Constant):
                return [f"{base_type}[{slice_value.value}]"]
            elif isinstance(slice_value, ast.Str):  # Python < 3.8
                return [f"{base_type}[{slice_value.s}]"]
            else:
                return [base_type]
        else:
            return [str(annotation)]
    else:
        return [str(annotation)]


def get_class_name(node):
    """Get the class name for a function, or 'global' if it's a module-level function"""
    current = node
    while hasattr(current, "parent"):
        current = current.parent
        if isinstance(current, ast.ClassDef):
            return current.name
    return "global"


def extract_function_signatures(file_path):
    """Extract function signatures with type information from a Python file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Try to fix f-string issues before parsing
        def fix_f_strings(code):
            """Try to fix common f-string parsing issues"""
            lines = code.split("\n")
            fixed_lines = []
            for i, line in enumerate(lines):
                # Handle problematic f-strings with unmatched parentheses
                if ('f"' in line or "f'" in line) and line.count("(") != line.count(
                    ")"
                ):
                    # Replace problematic f-strings with regular strings
                    line = line.replace('f"', '"').replace("f'", "'")
                    print(f"    - Fixed f-string on line {i+1}")
                fixed_lines.append(line)
            return "\n".join(fixed_lines)

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
                    return {}

        # Add parent references to nodes for class detection
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        function_signatures = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_name = get_class_name(node)
                function_key = f"{node.name}@{class_name}@"

                signature_info = []

                # Add arguments
                for arg in node.args.args:
                    arg_info = {
                        "category": "arg",
                        "name": arg.arg,
                        "type": get_type_annotation(arg.annotation),
                    }
                    signature_info.append(arg_info)

                # Add return type
                return_info = {
                    "category": "return",
                    "type": get_type_annotation(node.returns),
                }
                signature_info.append(return_info)

                function_signatures[function_key] = signature_info

        return function_signatures
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}


def generate_mapping():
    """Generate mapping between original and renamed function signatures"""
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

    print(
        f"Found {len(original_files)} original files and {len(renamed_files)} renamed files"
    )

    # Create comprehensive mapping with both original and renamed signatures
    all_mappings = {}

    for original_file in original_files:
        renamed_file = Path(renamed_dir) / original_file.name

        if not renamed_file.exists():
            print(f"Warning: No renamed version found for {original_file.name}")
            continue

        print(f"Processing: {original_file.name}")

        # Extract function signatures from both files
        original_signatures = extract_function_signatures(original_file)
        renamed_signatures = extract_function_signatures(renamed_file)

        print(f"  Original functions: {len(original_signatures)}")
        print(f"  Renamed functions: {len(renamed_signatures)}")

        # Create mapping for this file - list of function mappings
        file_mappings = []

        # Create function name mappings
        original_functions = list(original_signatures.keys())
        renamed_functions = list(renamed_signatures.keys())

        # Match functions by position (assuming same order)
        min_count = min(len(original_functions), len(renamed_functions))
        for i in range(min_count):
            original_key = original_functions[i]
            renamed_key = renamed_functions[i]

            # Only map if renamed function follows the expected pattern
            if (
                renamed_key.split("@")[0].startswith("func_")
                and len(renamed_key.split("@")[0]) == 13
            ):
                function_mapping = {
                    "original_func": original_key,
                    "renamed_func": renamed_key,
                }
                file_mappings.append(function_mapping)
                print(f"    {original_key} -> {renamed_key}")

        all_mappings[original_file.name] = file_mappings

    # Save comprehensive mappings to JSON file
    mapping_file = os.path.join(renamed_dir, "function_signature_mappings.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(all_mappings, f, indent=2)

    print(f"\nComprehensive mapping saved to: {mapping_file}")
    print(f"Total files processed: {len(all_mappings)}")

    # Print summary
    total_mappings = sum(len(mapping) for mapping in all_mappings.values())

    print(f"Total function mappings: {total_mappings}")


if __name__ == "__main__":
    generate_mapping()
