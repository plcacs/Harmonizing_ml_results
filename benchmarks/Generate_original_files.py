import os
import ast
import hashlib
import astunparse  # Install via: pip install astunparse

UN_TYPED_DIR = "original_benchmarks"
os.makedirs(UN_TYPED_DIR, exist_ok=True)

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
            f.write(code)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def traverse_and_process(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Only recurse into directories starting with 'bm_'
        if dirpath == root_dir:
            # Filter dirnames in-place to only keep 'bm_' dirs
            dirnames[:] = [d for d in dirnames if d.startswith("bm_")]
            continue  # Skip files in the root directory
        # Only process files if we're inside a 'bm_' directory
        if os.path.basename(dirpath).startswith("bm_"):
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    if UN_TYPED_DIR not in file_path:  # Avoid processing output files
                        process_py_file(file_path)

if __name__ == "__main__":
    traverse_and_process()
