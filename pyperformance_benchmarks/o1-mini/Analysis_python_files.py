import os
import ast
import json
from typing import Dict, Union

def analyze_python_file(file_path: str) -> Dict[str, Union[int, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        source: str = f.read()
    try:
        tree: ast.AST = ast.parse(source)
        func_count: int = 0
        param_count: int = 0

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
            "num_parameters": param_count
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_all_py_files(directory: str = ".") -> Dict[str, Union[Dict[str, Union[int, str]], str]]:
    results: Dict[str, Union[Dict[str, Union[int, str]], str]] = {}
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            abs_path: str = os.path.abspath(os.path.join(directory, filename))
            results[abs_path] = analyze_python_file(abs_path)
    return results

if __name__ == "__main__":
    output: Dict[str, Union[Dict[str, Union[int, str]], str]] = analyze_all_py_files()
    with open("analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
