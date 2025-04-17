import os
import ast
import hashlib
import astunparse  # Install via: pip install astunparse
from collections import defaultdict

UN_TYPED_DIR = "untyped_benchmarks"
os.makedirs(UN_TYPED_DIR, exist_ok=True)


def analyze_python_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
        func_count = 0
        param_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
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


class TypeRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        for arg in getattr(node.args, "kwonlyargs", []):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        # Convert annotated assignment to normal assignment
        return ast.Assign(
            targets=[node.target],
            value=node.value if node.value else ast.Constant(value=None),
            lineno=node.lineno,
            col_offset=node.col_offset,
        )


def hash_content(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def process_py_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    try:
        tree = ast.parse(code)
        tree = TypeRemover().visit(tree)
        ast.fix_missing_locations(tree)
        untyped_code = astunparse.unparse(tree)
        file_hash = hash_content(untyped_code)
        file_name = os.path.basename(file_path).replace(".py", f"_{file_hash}.py")
        output_path = os.path.join(UN_TYPED_DIR, file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(untyped_code)
        return output_path
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


import json


def traverse_and_process(root_dir="."):
    with open("short_filtered_paths.json", "r") as f:
        data = json.load(f)
    ordered_categories = ["50+", "30-50", "20-30", "10-20", "05-10"]
    total_files = set()
    results = {}
    group_number = 1
    member_count = 0
    group_dict = defaultdict(list)
    for category in ordered_categories:
        if category in data:
            for file_path in data[category]:
                if file_path not in total_files:
                    # print(f"Processing {file_path}")
                    results[file_path] = analyze_python_file(file_path)
                    output_path = process_py_file(file_path)
                    total_files.add(file_path)
                    member_count += 1
                    group_dict[group_number].append(file_path)
                    if member_count >= 150:
                        group_number += 1
                        member_count = 0

    print("Total File count", len(total_files))
    print("Total group count: ", len(group_dict.keys()))
    # for path in data.values():
    # print(path)
    #    process_py_file(path)
    """for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                if UN_TYPED_DIR not in file_path:  # Avoid processing output files
                    process_py_file(file_path)"""


if __name__ == "__main__":
    traverse_and_process()
