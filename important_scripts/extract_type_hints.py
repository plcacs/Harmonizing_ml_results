#!/usr/bin/env python3
"""
Script to extract type hint information from files mentioned in the typing analysis JSON.
Compares type annotations between old (LLM-generated) and new (cloned repository) files.
Focuses on function parameters and return types only.
"""

import json
import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import argparse


@dataclass
class TypeHintInfo:
    """Data class to store type hint information for a file."""

    file_path: str
    total_functions: int
    typed_functions: int
    total_parameters: int
    typed_parameters: int
    function_details: List[Dict[str, Any]]
    parse_errors: List[str]


class TypeHintExtractor:
    """Extract type hint information from Python files."""

    def __init__(self, base_dir: str = "Harmonizing_ml_results"):
        self.base_dir = Path(base_dir)

    def normalize_path(self, file_path: str) -> Path:
        """Convert Windows absolute path to relative path from base directory."""
        # Extract path starting from Harmonizing_ml_results
        if "Harmonizing_ml_results" in file_path:
            parts = file_path.split("Harmonizing_ml_results")
            if len(parts) > 1:
                relative_path = parts[1].lstrip("\\").replace("\\", "/")
                return self.base_dir / relative_path
        return Path(file_path)

    def extract_type_hints_from_file(self, file_path: Path) -> TypeHintInfo:
        """Extract type hint information from a Python file."""
        info = TypeHintInfo(
            file_path=str(file_path),
            total_functions=0,
            typed_functions=0,
            total_parameters=0,
            typed_parameters=0,
            function_details=[],
            parse_errors=[],
        )

        if not file_path.exists():
            info.parse_errors.append(f"File not found: {file_path}")
            return info

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            self._analyze_ast(tree, info)

        except SyntaxError as e:
            info.parse_errors.append(f"Syntax error: {e}")
        except Exception as e:
            info.parse_errors.append(f"Error parsing file: {e}")

        return info

    def _analyze_ast(self, tree: ast.AST, info: TypeHintInfo):
        """Analyze AST to extract type hint information."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._analyze_function(node, info)
            elif isinstance(node, ast.AsyncFunctionDef):
                self._analyze_function(node, info)

    def _analyze_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], info: TypeHintInfo
    ):
        """Analyze function definition for type hints."""
        info.total_functions += 1

        func_info = {
            "name": node.name,
            "line": node.lineno,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "has_return_annotation": node.returns is not None,
            "return_annotation": self._get_annotation_string(node.returns),
            "parameters": [],
        }

        typed_params = 0
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "has_annotation": arg.annotation is not None,
                "annotation": self._get_annotation_string(arg.annotation),
            }
            func_info["parameters"].append(param_info)

            if arg.annotation is not None:
                typed_params += 1

        info.total_parameters += len(node.args.args)
        info.typed_parameters += typed_params

        if node.returns is not None or typed_params > 0:
            info.typed_functions += 1

        info.function_details.append(func_info)

    def _get_annotation_string(self, annotation) -> Optional[str]:
        """Convert annotation to string representation."""
        if annotation is None:
            return None
        try:
            return ast.unparse(annotation)
        except:
            return str(annotation)


def process_typing_analysis(
    input_file: str, output_file: str, base_dir: str = "Harmonizing_ml_results"
):
    """Process the typing analysis JSON file and extract detailed type hint information."""

    # Load the analysis file
    with open(input_file, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    extractor = TypeHintExtractor(base_dir)
    results = {
        "summary": {
            "total_entries": len(analysis_data),
            "processed_files": 0,
            "errors": [],
        },
        "file_analysis": {},
    }

    for signature, data in analysis_data.items():
        entry_results = {
            "signature": signature,
            "old_file": None,
            "new_file": None,
            "comparison": {},
        }

        # Process old file
        if "old" in data and "file" in data["old"]:
            old_path = extractor.normalize_path(data["old"]["file"])
            try:
                old_info = extractor.extract_type_hints_from_file(old_path)
                entry_results["old_file"] = asdict(old_info)
                results["summary"]["processed_files"] += 1
            except Exception as e:
                results["summary"]["errors"].append(
                    f"Error processing old file {old_path}: {e}"
                )

        # Process new file
        if "new" in data and "file" in data["new"]:
            new_path = extractor.normalize_path(data["new"]["file"])
            try:
                new_info = extractor.extract_type_hints_from_file(new_path)
                entry_results["new_file"] = asdict(new_info)
                results["summary"]["processed_files"] += 1
            except Exception as e:
                results["summary"]["errors"].append(
                    f"Error processing new file {new_path}: {e}"
                )

        # Add comparison metrics
        if entry_results["old_file"] and entry_results["new_file"]:
            old = entry_results["old_file"]
            new = entry_results["new_file"]

            entry_results["comparison"] = {
                "function_improvement": new["typed_functions"] - old["typed_functions"],
                "parameter_improvement": new["typed_parameters"]
                - old["typed_parameters"],
                "function_typing_ratio_old": old["typed_functions"]
                / max(1, old["total_functions"]),
                "function_typing_ratio_new": new["typed_functions"]
                / max(1, new["total_functions"]),
                "parameter_typing_ratio_old": old["typed_parameters"]
                / max(1, old["total_parameters"]),
                "parameter_typing_ratio_new": new["typed_parameters"]
                / max(1, new["total_parameters"]),
            }

        results["file_analysis"][signature] = entry_results

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Analysis complete!")
    print(f"Total entries processed: {results['summary']['total_entries']}")
    print(f"Files successfully analyzed: {results['summary']['processed_files']}")
    print(f"Errors encountered: {len(results['summary']['errors'])}")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract type hint information from files in typing analysis"
    )
    parser.add_argument("input_file", help="Path to the typing analysis JSON file")
    parser.add_argument(
        "output_file", help="Path to save the detailed analysis results"
    )
    parser.add_argument(
        "--base-dir",
        default="Harmonizing_ml_results",
        help="Base directory for file paths (default: Harmonizing_ml_results)",
    )

    args = parser.parse_args()

    process_typing_analysis(args.input_file, args.output_file, args.base_dir)


if __name__ == "__main__":
    main()
