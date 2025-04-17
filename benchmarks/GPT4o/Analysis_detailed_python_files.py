import os
import ast
import json
from typing import Dict, Any, Union, List

def analyze_python_file(file_path: str) -> Dict[str, Union[int, str, List[Dict[str, Any]]]]:
    with open(file_path, "r", encoding="utf-8") as f:
        source: str = f.read()
    try:
        tree: ast.Module = ast.parse(source)
        func_count: int = 0
        param_count: int = 0
        annotations: List[Dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_count += 1
                param_count += len(node.args.args) + len(getattr(node.args, "kwonlyargs", []))
                if node.args.vararg:
                    param_count += 1
                if node.args.kwarg:
                    param_count += 1

                param_hints = {
                    arg.arg: ast.unparse(arg.annotation) if arg.annotation else None
                    for arg in node.args.args + node.args.kwonlyargs
                }
                if node.args.vararg:
                    param_hints[node.args.vararg.arg] = ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else None
                if node.args.kwarg:
                    param_hints[node.args.kwarg.arg] = ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else None

                annotations.append({
                    "function": node.name,
                    "parameter_hints": param_hints,
                    "return_hint": ast.unparse(node.returns) if node.returns else None
                })

        return {
            "num_lines": len(source.splitlines()),
            "num_functions": func_count,
            "num_parameters": param_count,
            "type_hints": annotations
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_all_py_files(directory: str = ".") -> Dict[str, Dict[str, Union[int, str]]]:
    results: Dict[str, Dict[str, Union[int, str]]] = {}
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            abs_path: str = os.path.abspath(os.path.join(directory, filename))
            results[abs_path] = analyze_python_file(abs_path)
    return results

if __name__ == "__main__":
    output: Dict[str, Dict[str, Union[int, str]]] = analyze_all_py_files()
    with open("analysis_detailed_gpt4o.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
