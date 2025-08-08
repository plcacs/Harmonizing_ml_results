import ast
import os
import random
import string
from pathlib import Path

class FunctionRenamer(ast.NodeTransformer):
    def __init__(self):
        self.function_mapping = {}
        self.builtin_functions = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
            'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
            'abs', 'max', 'min', 'sum', 'all', 'any', 'isinstance', 'hasattr',
            'getattr', 'setattr', 'delattr', 'super', 'property', 'staticmethod',
            'classmethod', 'open', 'input', 'eval', 'exec', 'compile', 'hash',
            'id', 'type', 'dir', 'vars', 'locals', 'globals', 'help', 'repr'
        }
    
    def generate_random_name(self, original_name):
        """Generate a random function name"""
        prefix = "func_"
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{prefix}{suffix}"
    
    def visit_FunctionDef(self, node):
        """Rename function definitions"""
        if node.name not in self.builtin_functions and not node.name.startswith('__'):
            if node.name not in self.function_mapping:
                self.function_mapping[node.name] = self.generate_random_name(node.name)
            node.name = self.function_mapping[node.name]
        return self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Rename async function definitions"""
        if node.name not in self.builtin_functions and not node.name.startswith('__'):
            if node.name not in self.function_mapping:
                self.function_mapping[node.name] = self.generate_random_name(node.name)
            node.name = self.function_mapping[node.name]
        return self.generic_visit(node)
    
    def visit_Call(self, node):
        """Rename function calls"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.function_mapping:
                node.func.id = self.function_mapping[func_name]
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls (e.g., obj.method())
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                if obj_name in self.function_mapping:
                    # This is a method call on a renamed function
                    node.func.value.id = self.function_mapping[obj_name]
        return self.generic_visit(node)

def rename_functions_in_file(file_path, output_dir):
    """Rename all functions in a single file and save to output directory"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Preprocess to handle some common f-string issues
        def fix_f_strings(code):
            """Try to fix common f-string parsing issues"""
            lines = code.split('\n')
            fixed_lines = []
            for i, line in enumerate(lines):
                # Handle problematic f-strings with unmatched parentheses
                if ('f"' in line or "f'" in line) and line.count('(') != line.count(')'):
                    # Replace problematic f-strings with regular strings
                    line = line.replace('f"', '"').replace("f'", "'")
                    print(f"    - Fixed f-string on line {i+1}: {line.strip()}")
                fixed_lines.append(line)
            return '\n'.join(fixed_lines)
        
        # Try to parse the source code with different approaches
        tree = None
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Try with fixed f-strings
            try:
                fixed_source = fix_f_strings(source_code)
                tree = ast.parse(fixed_source)
            except SyntaxError:
                # Try with a more aggressive f-string fix
                try:
                    # Remove all f-string prefixes that might cause issues
                    aggressive_fixed = source_code.replace('f"', '"').replace("f'", "'")
                    tree = ast.parse(aggressive_fixed)
                    print(f"    - Applied aggressive f-string fix")
                except SyntaxError:
                    # Try with compile mode to handle some edge cases
                    try:
                        compile(source_code, '<string>', 'exec')
                        # If compile succeeds, try parsing with different mode
                        tree = ast.parse(source_code, mode='exec')
                    except SyntaxError as syntax_error:
                        print(f"  - Syntax error in {file_path.name}: {syntax_error}")
                        print(f"  - Skipping file due to syntax error")
                        # Copy the original file to output directory without changes
                        output_file = os.path.join(output_dir, file_path.name)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(source_code)
                        return 0
        
        # Create renamer and transform the AST
        renamer = FunctionRenamer()
        new_tree = renamer.visit(tree)
        
        # Convert back to source code
        import astor
        new_source = astor.to_source(new_tree)
        
        # Create output file path
        output_file = os.path.join(output_dir, file_path.name)
        
        # Write the modified code to output directory
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_source)
        
        return len(renamer.function_mapping)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Copy the original file to output directory without changes
        output_file = os.path.join(output_dir, file_path.name)
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(output_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        return 0

def process_all_files():
    """Process all Python files in the target directory"""
    source_dir = "Hundrad_original_typed_benchmarks"
    output_dir = "Hundrad_renamed_benchmarks"
    
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist. Please run collect_top_100_files.py first.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    python_files = list(Path(source_dir).glob("*.py"))
    
    if not python_files:
        print(f"No Python files found in {source_dir}")
        return
    
    print(f"Found {len(python_files)} Python files to process")
    print(f"Original files preserved in: {source_dir}")
    print(f"Renamed files saved to: {output_dir}")
    
    total_functions_renamed = 0
    processed_files = 0
    
    for file_path in python_files:
        print(f"Processing: {file_path.name}")
        functions_renamed = rename_functions_in_file(file_path, output_dir)
        total_functions_renamed += functions_renamed
        processed_files += 1
        print(f"  - Renamed {functions_renamed} functions")
    
    print(f"\nCompleted! Processed {processed_files} files, renamed {total_functions_renamed} functions total")
    print(f"Original files: {source_dir}")
    print(f"Renamed files: {output_dir}")

if __name__ == "__main__":
    process_all_files() 