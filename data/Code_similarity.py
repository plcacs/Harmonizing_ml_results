import ast
import os
import difflib
from pathlib import Path
from itertools import product
from ast import unparse
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
# Directory groups
deepseek_dirs = ['deep_seek', 'deep_seek2']
o1mini_dirs = ['o1-mini', 'o1-mini2', 'o1-mini3']
gpt4o_dirs = ['GPT4o', 'GPT4o2']

# AST Transformer to strip type annotations
class TypeAnnotationRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        return ast.Assign(targets=[node.target], value=node.value or ast.Constant(value=None))

def strip_annotations(code):
    try:
        tree = ast.parse(code)
        tree = TypeAnnotationRemover().visit(tree)
        return unparse(tree)
    except Exception:
        return ""

def get_py_files(dirs):
    return [f for d in dirs for f in Path(d).rglob("*.py")]

def read_and_strip(path):
    try:
        with open(path) as f:
            return strip_annotations(f.read())
    except:
        return ""

def compare_all_triplets(deepseek_files, o1mini_files, gpt4o_files, threshold=0.9):
    results = []
    for f1, f2, f3 in product(deepseek_files, o1mini_files, gpt4o_files):
        c1 = read_and_strip(f1)
        c2 = read_and_strip(f2)
        r12 = difflib.SequenceMatcher(None, c1, c2).ratio()
        if r12 < threshold:
            continue

        c3 = read_and_strip(f3)
        r23 = difflib.SequenceMatcher(None, c2, c3).ratio()
        if r23 < threshold:
            continue

        r13 = difflib.SequenceMatcher(None, c1, c3).ratio()
        if r13 < threshold:
            continue
        print()(f"Match: {f1.name} ↔ {f2.name} ↔ {f3.name} | Sim: {r12:.2f}, {r23:.2f}, {r13:.2f}")
        results.append((f1.name, f2.name, f3.name, r12, r23, r13))
    return results

if __name__ == "__main__":
    ds_files = get_py_files(deepseek_dirs)
    o1_files = get_py_files(o1mini_dirs)
    g4_files = get_py_files(gpt4o_dirs)

    print(f"Comparing {len(ds_files)} × {len(o1_files)} × {len(g4_files)} = {len(ds_files)*len(o1_files)*len(g4_files)} triplets")
    groups = compare_all_triplets(ds_files, o1_files, g4_files)
    
    for g in groups:
        print(f"Match: {g[0]} ↔ {g[1]} ↔ {g[2]} | Sim: {g[3]:.2f}, {g[4]:.2f}, {g[5]:.2f}")
