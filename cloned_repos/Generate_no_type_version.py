#!/usr/bin/env python3
import os
import ast
import argparse
import json
import sys
import warnings
import hashlib
from collections import defaultdict

# Suppress invalid escape sequence warnings when parsing source files
warnings.filterwarnings("ignore", category=SyntaxWarning)

UN_TYPED_DIR = "untyped_benchmarks"
os.makedirs(UN_TYPED_DIR, exist_ok=True)

class RemoveTypeHints(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        # Visit body first to remove annotations inside class
        self.generic_visit(node)
        # If class body is empty after removals, insert a pass
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        # Remove return annotation
        node.returns = None
        # Remove parameter annotations
        for arg in node.args.args + getattr(node.args, "kwonlyargs", []):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node):
        # Convert "x: T = value" to "x = value"
        if node.value:
            new_node = ast.Assign(
                targets=[node.target],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset
            )
            return ast.copy_location(new_node, node)
        return None

    def visit_arg(self, node):
        node.annotation = None
        return node


def analyze_python_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=file_path)
        func_count = 0
        param_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_count += 1
                param_count += len(node.args.args)
                param_count += len(getattr(node.args, "kwonlyargs", []))
                if node.args.vararg:
                    param_count += 1
                if node.args.kwarg:
                    param_count += 1
        return {
            "num_lines": len(source.splitlines()),
            "num_functions": func_count,
            "num_parameters": param_count,
        }
    except Exception as e:
        return {"error": str(e)}


def process_py_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code, filename=file_path)
        transformer = RemoveTypeHints()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        try:
            untyped_code = ast.unparse(tree)
        except AttributeError:
            import astunparse
            untyped_code = astunparse.unparse(tree)
        # Compute a 6-digit hash suffix from content
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        hash6 = hashlib.md5(untyped_code.encode('utf-8')).hexdigest()[:6]
        new_filename = f"{base_name}_{hash6}.py"
        output_path = os.path.join(UN_TYPED_DIR, new_filename)
        with open(output_path, "w", encoding="utf-8") as outf:
            outf.write(untyped_code)
        return output_path
    except SyntaxError as e:
        print(f"Syntax error in '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


def traverse_and_process():
    with open("filtered_python_files.json", "r") as f:
        data = json.load(f)
    ordered_categories = ["50+", "30-50", "20-30", "10-20", "05-10"]
    total_files = set()
    results = {}
    group_number = 1
    member_count = 0
    group_dict = defaultdict(list)
    for category in ordered_categories:
        for file_path in data.get(category, []):
            if file_path in total_files:
                continue
            results[file_path] = analyze_python_file(file_path)
            out = process_py_file(file_path)
            if out:
                total_files.add(file_path)
                group_dict[group_number].append(out)
                member_count += 1
                if member_count >= 150:
                    group_number += 1
                    member_count = 0
    print("Total File count", len(total_files))
    print("Total group count:", len(group_dict))
    with open("grouped_file_paths.json", "w") as f:
        json.dump(group_dict, f, indent=2)
    with open("File_analysis_result.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Remove type hints from Python files."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a Python file. If omitted, uses filtered_python_files.json."
    )
    args = parser.parse_args()

    if args.input_path:
        print(f"Processing single file: {args.input_path}")
        print(f"Analysis: {analyze_python_file(args.input_path)}")
        out = process_py_file(args.input_path)
        print(f"Untyped file saved to: {out}")
    else:
        traverse_and_process()

if __name__ == "__main__":
    main()
