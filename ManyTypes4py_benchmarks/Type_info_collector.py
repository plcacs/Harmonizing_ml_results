import os
import ast
import json
from typing import Any, Dict, List, Tuple


def extract_annotation(annotation: ast.AST) -> str:
    if annotation is None:
        return ""
    try:
        return ast.unparse(annotation)
    except Exception:
        return ast.dump(annotation)


def process_function(
    node: ast.FunctionDef, class_name: str = "", filename: str = ""
) -> Dict[str, Any]:
    func_key = f"{node.name}@{class_name or 'global'}@{filename}"
    result = []

    for arg in node.args.args:
        annotation = extract_annotation(arg.annotation)
        result.append({"category": "arg", "name": arg.arg, "type": [annotation]})

    if node.returns:
        return_annotation = extract_annotation(node.returns)
        result.append({"category": "return", "type": [return_annotation]})

    return {func_key: result}


def process_file(filepath: str) -> Dict[str, Any]:
    # Try different encodings to handle Unicode issues
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    source = None

    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                source = f.read()
            break
        except UnicodeDecodeError:
            continue

    if source is None:
        print(f"Failed to read {filepath} with any encoding")
        return {}

    try:
        tree = ast.parse(source, filename=filepath)
    except Exception as e:
        print(f"Failed to parse {filepath}: {e}")
        return {}

    annotations = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            annotations.update(process_function(node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    annotations.update(process_function(item, node.name))
    return annotations


def collect_annotations(root_dir: str, output_json: str):
    all_annotations = {}
    file_count = 0
    error_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                fullpath = os.path.join(dirpath, filename)
                file_count += 1

                try:
                    annots = process_file(fullpath)
                    if annots:  # Only add if we got some annotations
                        all_annotations[filename] = annots
                    else:
                        # Still count files that were processed but had no annotations
                        all_annotations[filename] = {}
                except Exception as e:
                    # print(f"Error processing {fullpath}: {e}")
                    error_count += 1
                    # Still add the file to the output with empty annotations
                    all_annotations[filename] = {}

    with open(output_json, "w") as f:
        json.dump(all_annotations, f, indent=2)

    print(f"Annotations saved to {output_json}")
    print(f"Number of files processed: {file_count}")
    print(f"Number of files with errors: {error_count}")


# Example usage
# collect_annotations("/path/to/python/codebase", "output_annotations.json")
collect_annotations(
    "original_files", "Type_info_collector/Type_info_original_files.json"
)
# collect_annotations(
#     "claude3_sonnet_1st_run",
#     "Type_info_collector/Type_info_claude3_sonnet_1st_run_benchmarks.json",
# )
