"""
Generate intra-file call graph metrics for every Python file in a directory.

Usage:
    python generate_callgraph.py <input_directory> [output_json]

    input_directory : path to a folder containing .py files (flat or nested)
    output_json     : optional output path (default: callgraph_metrics.json
                      saved in this script's directory)

Metrics per file:
    - num_functions        : total function/method definitions
    - num_call_edges       : total caller->callee edges (intra-file only)
    - max_fan_out          : max calls made by a single function
    - avg_fan_out          : average calls per function
    - max_fan_in           : max times a function is called by others
    - avg_fan_in           : average times each function is called
    - max_call_depth       : longest call chain (DFS-based)
    - has_recursion         : whether any function calls itself
    - num_connected_components : isolated clusters of functions
    - num_classes          : number of class definitions
"""

import ast
import json
import os
import sys
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class CallGraphVisitor(ast.NodeVisitor):
    """Walk an AST to extract function definitions and intra-file call edges."""

    def __init__(self):
        self.functions = set()
        self.classes = set()
        self.edges = []
        self._current_func = None

    def visit_ClassDef(self, node):
        self.classes.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        func_name = node.name
        self.functions.add(func_name)
        parent = self._current_func
        self._current_func = func_name
        self.generic_visit(node)
        self._current_func = parent

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if self._current_func is None:
            self.generic_visit(node)
            return

        callee = self._resolve_callee(node.func)
        if callee:
            self.edges.append((self._current_func, callee))
        self.generic_visit(node)

    @staticmethod
    def _resolve_callee(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


def compute_metrics(functions, edges, num_classes):
    """Derive call graph metrics from raw functions and edges."""
    func_set = set(functions)
    intra_edges = [(a, b) for a, b in edges if b in func_set]

    fan_out = defaultdict(int)
    fan_in = defaultdict(int)
    adjacency = defaultdict(set)

    for caller, callee in intra_edges:
        fan_out[caller] += 1
        fan_in[callee] += 1
        adjacency[caller].add(callee)

    has_recursion = any(caller == callee for caller, callee in intra_edges)

    max_fan_out = max(fan_out.values()) if fan_out else 0
    avg_fan_out = (
        round(sum(fan_out.values()) / len(func_set), 2) if func_set else 0
    )
    max_fan_in = max(fan_in.values()) if fan_in else 0
    avg_fan_in = (
        round(sum(fan_in.values()) / len(func_set), 2) if func_set else 0
    )

    max_depth = _max_call_depth(adjacency, func_set)
    num_components = _count_components(func_set, intra_edges)

    return {
        "num_functions": len(func_set),
        "num_call_edges": len(intra_edges),
        "max_fan_out": max_fan_out,
        "avg_fan_out": avg_fan_out,
        "max_fan_in": max_fan_in,
        "avg_fan_in": avg_fan_in,
        "max_call_depth": max_depth,
        "has_recursion": has_recursion,
        "num_connected_components": num_components,
        "num_classes": num_classes,
    }


def _max_call_depth(adjacency, func_set):
    """DFS-based longest call chain length."""
    cache = {}

    def dfs(node, visited):
        if node in cache:
            return cache[node]
        if node in visited:
            return 0
        visited.add(node)
        depth = 0
        for neighbor in adjacency.get(node, []):
            if neighbor in func_set:
                depth = max(depth, 1 + dfs(neighbor, visited))
        visited.discard(node)
        cache[node] = depth
        return depth

    return max((dfs(f, set()) for f in func_set), default=0)


def _count_components(func_set, edges):
    """Count connected components in the undirected version of the call graph."""
    if not func_set:
        return 0

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited = set()
    components = 0
    for f in func_set:
        if f not in visited:
            components += 1
            stack = [f]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(adj[node] - visited)
    return components


def analyze_file(filepath):
    """Parse a single Python file and return its call graph metrics."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, ValueError):
        return None

    visitor = CallGraphVisitor()
    visitor.visit(tree)

    return compute_metrics(visitor.functions, visitor.edges, len(visitor.classes))


def process_directory(input_dir):
    """Walk a directory, analyze every .py file, return results dict."""
    results = {}
    py_files = []

    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))

    total = len(py_files)
    print(f"Found {total} Python files in {input_dir}")

    for i, filepath in enumerate(sorted(py_files), 1):
        metrics = analyze_file(filepath)
        rel_path = os.path.relpath(filepath, input_dir)
        if metrics is not None:
            results[rel_path] = metrics
        else:
            results[rel_path] = {"error": "parse_failed"}

        if i % 200 == 0 or i == total:
            print(f"  Processed {i}/{total} files")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_callgraph.py <input_directory> [output_json]")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.join(SCRIPT_DIR, "callgraph_metrics.json")

    results = process_directory(input_dir)

    parsed = sum(1 for v in results.values() if "error" not in v)
    failed = len(results) - parsed
    print(f"\nDone: {parsed} parsed, {failed} failed, {len(results)} total")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
