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


def parse_with_recovery(source: str, filename: str, max_skips: int = 100):
    """Attempt to parse source; on SyntaxError, blank offending line and retry.

    Returns a tuple of (ast.AST, List[int]) containing the parsed tree and the
    list of line numbers that were skipped. Raises if recovery exceeds max_skips
    or the error lacks a usable line number.
    """
    skipped_lines: List[int] = []
    current_source = source
    while True:
        try:
            return ast.parse(current_source, filename=filename), skipped_lines
        except SyntaxError as e:
            line_number = getattr(e, "lineno", None)
            if line_number is None or len(skipped_lines) >= max_skips:
                raise
            lines = current_source.splitlines()
            if 1 <= line_number <= len(lines):
                # Blank the offending line and retry
                lines[line_number - 1] = ""
                skipped_lines.append(line_number)
                current_source = "\n".join(lines)
            else:
                raise


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
        return {}

    try:
        tree, _skipped = parse_with_recovery(source, filepath, max_skips=200)
    except Exception:
        return {}

    annotations = {}
    processed_functions = set()  # Track processed functions to avoid duplicates

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Only process if this function hasn't been processed yet
            if node not in processed_functions:
                annotations.update(process_function(node))
                processed_functions.add(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # Process class methods and mark as processed
                    annotations.update(process_function(item, node.name))
                    processed_functions.add(item)
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

    num_non_empty = sum(1 for v in all_annotations.values() if v)
    num_empty = sum(1 for v in all_annotations.values() if not v)
    print(f"non_empty: {num_non_empty}")
    print(f"empty: {num_empty}")


# Example usage
# collect_annotations("/path/to/python/codebase", "output_annotations.json")
# collect_annotations(
#     "original_files", "Type_info_collector/Type_info_original_files.json"
# )
"""collect_annotations(
    "claude3_sonnet_1st_run",
    "Type_info_collector/Type_info_claude3_sonnet_1st_run_benchmarks.json",
)
collect_annotations(
    "o3_mini_1st_run",
    "Type_info_collector/Type_info_o3_mini_1st_run_benchmarks.json",
)
collect_annotations(
    "gpt4o",
    "Type_info_collector/Type_info_gpt4o_benchmarks.json",
)"""
collect_annotations(
    "claude3_sonnet_user_annotated",
    "Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_user_annotated_benchmarks.json",
)
collect_annotations(
    "deepseek_user_annotated",
    "Type_info_collector/Type_info_LLMS/Type_info_deepseek_user_annotated_benchmarks.json",
)
collect_annotations(
    "o3_mini_user_annotated",
    "Type_info_collector/Type_info_LLMS/Type_info_o3_mini_user_annotated_benchmarks.json",
)
