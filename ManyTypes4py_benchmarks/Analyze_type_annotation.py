import os
import json
import ast
from collections import Counter
from typing import Any, Dict, List, Tuple

def get_annotation_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Subscript):
        return get_annotation_name(node.value)
    elif isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Call):
        return get_annotation_name(node.func)
    return "unknown"

def count_nested_depth(node, level=1):
    if isinstance(node, ast.Subscript):
        return max(count_nested_depth(node.slice, level + 1), level)
    elif isinstance(node, ast.Tuple):
        return max(count_nested_depth(elt, level + 1) for elt in node.elts)
    elif isinstance(node, ast.List):
        return max(count_nested_depth(elt, level + 1) for elt in node.elts)
    return level

def analyze_function_annotations(func: ast.FunctionDef) -> Tuple[int, Counter, List[str], List[int]]:
    counts = Counter()
    examples = []
    nested_depths = []
    annotated_count = 0

    for arg in func.args.args + func.args.kwonlyargs:
        if arg.annotation:
            annotated_count += 1
            name = get_annotation_name(arg.annotation)
            counts[name] += 1
            if name in ("Union", "Optional", "Literal", "Any", "Callable", "List", "Dict", "Set", "Tuple"):
                examples.append(ast.unparse(arg.annotation))
            nested_depths.append(count_nested_depth(arg.annotation))

    return annotated_count, counts, examples, nested_depths

def analyze_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    lines = len(source.splitlines())
    total_funcs = 0
    funcs_with_ann = 0
    total_params = 0
    total_annots = 0
    type_counter = Counter()
    nested_examples = []
    nested_depths = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            total_funcs += 1
            param_count = len(node.args.args + node.args.kwonlyargs)
            total_params += param_count
            annotated, annots, examples, depths = analyze_function_annotations(node)
            if annotated > 0:
                funcs_with_ann += 1
            total_annots += annotated
            type_counter.update(annots)
            #nested_examples.extend(examples)
            nested_depths.extend(depths)

    result = {
        "filename": os.path.basename(filepath),
        "total_lines": lines,
        "function_summary": {
            "total_functions": total_funcs,
            "functions_with_annotations": funcs_with_ann,
            "total_parameters": total_params,
            "annotated_parameters": total_annots,
            "annotation_coverage_percent": round((total_annots / total_params * 100) if total_params else 0, 2)
        },
        "type_annotation_stats": {
            k: {
                "count": v,
                "percent": round((v / total_annots * 100), 2) if total_annots else 0
            }
            for k, v in type_counter.items()
        },
        "nested_type_depth": {
            "max": max(nested_depths) if nested_depths else 0,
            "average": round(sum(nested_depths) / len(nested_depths), 2) if nested_depths else 0,
            #"examples": nested_examples[:5]
        }
    }
    return result

def analyze_directory(directory: str, output_path: str):
    results = []
    error_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                try:
                    result = analyze_file(full_path)
                    results.append(result)
                except Exception as e:
                    error_files.append({"file": file, "error": str(e)})
                    continue
    
    # Save successful results
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)
    
    # Save error log
    error_path = output_path.replace(".json", "_errors.json")
    with open(error_path, "w", encoding="utf-8") as error_file:
        json.dump(error_files, error_file, indent=2)
    
    print(f"Processed {len(results)} files successfully")
    print(f"Encountered errors in {len(error_files)} files")
    print(f"Error log saved to: {error_path}")
    
    # Print total instances from output file
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Total instances in output file: {len(data)}")

analyze_directory("gpt4o", "mypy_results/gpt4o_benchmarks_type_annotation_stats.json")