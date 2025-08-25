import os
import ast
import hashlib
import astunparse
from typing import Optional
from ast import NodeTransformer, Node, FunctionDef, AnnAssign, Assign, Constant
from ast import parse, fix_missing_locations, unparse
from hashlib import md5

UN_TYPED_DIR: str = 'untyped_benchmarks'
os.makedirs(UN_TYPED_DIR, exist_ok=True)

class TypeRemover(NodeTransformer):

    def visit_FunctionDef(self, node: FunctionDef) -> Optional[Node]:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        for arg in getattr(node.args, 'kwonlyargs', []):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: AnnAssign) -> Assign:
        return Assign(targets=[node.target], value=(node.value if node.value else Constant(value=None)), lineno=node.lineno, col_offset=node.col_offset)

def hash_content(content: str) -> str:
    return md5(content.encode('utf-8')).hexdigest()

def process_py_file(file_path: str) -> None:
    with open(file_path, 'r', encoding='utf-8') as f:
        code: str = f.read()
    try:
        tree: ast.AST = parse(code)
        tree = TypeRemover().visit(tree)
        fix_missing_locations(tree)
        untyped_code: str = unparse(tree)
        file_hash: str = hash_content(untyped_code)
        file_name: str = os.path.basename(file_path).replace('.py', f'_{file_hash}.py')
        output_path: str = os.path.join(UN_TYPED_DIR, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(untyped_code)
    except Exception as e:
        print(f'Failed to process {file_path}: {e}')

def traverse_and_process(root_dir: str = '.') -> None:
    for (dirpath, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path: str = os.path.join(dirpath, filename)
                if (UN_TYPED_DIR not in file_path):
                    process_py_file(file_path)

if (__name__ == '__main__'):
    traverse_and_process()
