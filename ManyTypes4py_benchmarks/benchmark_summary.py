import os
import ast
import warnings
from pathlib import Path
from typing import Dict, List, Optional


def analyze_python_file(file_path: str) -> Dict:
    """Analyze a single Python file and return statistics."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=file_path)
        
        line_count = len(source.splitlines())
        func_count = 0
        param_count = 0
        param_annotated_count = 0
        return_count = 0
        return_annotated_count = 0
        class_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_count += 1
                # Count parameters and annotated parameters
                for arg in node.args.args:
                    param_count += 1
                    if arg.annotation is not None:
                        param_annotated_count += 1
                # Count keyword-only arguments
                for arg in getattr(node.args, "kwonlyargs", []):
                    param_count += 1
                    if arg.annotation is not None:
                        param_annotated_count += 1
                # Count vararg and kwarg
                if node.args.vararg:
                    param_count += 1
                    if node.args.vararg.annotation is not None:
                        param_annotated_count += 1
                if node.args.kwarg:
                    param_count += 1
                    if node.args.kwarg.annotation is not None:
                        param_annotated_count += 1
                # Count return annotations
                if node.returns is not None:
                    return_count += 1
                    return_annotated_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
        
        return {
            "line_count": line_count,
            "func_count": func_count,
            "param_count": param_count,
            "param_annotated_count": param_annotated_count,
            "return_count": return_count,
            "return_annotated_count": return_annotated_count,
            "class_count": class_count,
            "error": None
        }
    except Exception as e:
        return {
            "line_count": 0,
            "func_count": 0,
            "param_count": 0,
            "param_annotated_count": 0,
            "return_count": 0,
            "return_annotated_count": 0,
            "class_count": 0,
            "error": str(e)
        }


def calculate_stats(values: List[int]) -> Dict:
    """Calculate min, max, and average for a list of values."""
    if not values:
        return {"min": 0, "max": 0, "avg": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values)
    }


def analyze_benchmarks(input_dir: str) -> None:
    """Analyze all Python files in the input directory and print summary."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    results = []
    error_count = 0
    
    # Analyze all Python files
    for file_path in input_path.glob("*.py"):
        result = analyze_python_file(str(file_path))
        results.append(result)
        if result["error"]:
            error_count += 1
    
    if not results:
        print(f"No Python files found in {input_dir}")
        return
    
    # Extract statistics
    line_counts = [r["line_count"] for r in results if not r["error"]]
    func_counts = [r["func_count"] for r in results if not r["error"]]
    param_counts = [r["param_count"] for r in results if not r["error"]]
    param_annotated_counts = [r["param_annotated_count"] for r in results if not r["error"]]
    return_counts = [r["return_count"] for r in results if not r["error"]]
    return_annotated_counts = [r["return_annotated_count"] for r in results if not r["error"]]
    class_counts = [r["class_count"] for r in results if not r["error"]]
    
    # Calculate statistics
    line_stats = calculate_stats(line_counts)
    func_stats = calculate_stats(func_counts)
    param_stats = calculate_stats(param_counts)
    param_annotated_stats = calculate_stats(param_annotated_counts)
    return_stats = calculate_stats(return_counts)
    return_annotated_stats = calculate_stats(return_annotated_counts)
    class_stats = calculate_stats(class_counts)
    
    # Additional statistics
    total_files = len(results)
    total_lines = sum(line_counts)
    total_functions = sum(func_counts)
    total_parameters = sum(param_counts)
    total_param_annotated = sum(param_annotated_counts)
    total_returns = sum(return_counts)
    total_return_annotated = sum(return_annotated_counts)
    total_classes = sum(class_counts)
    
    files_with_functions = sum(1 for r in results if not r["error"] and r["func_count"] > 0)
    files_with_classes = sum(1 for r in results if not r["error"] and r["class_count"] > 0)
    files_with_returns = sum(1 for r in results if not r["error"] and r["return_count"] > 0)
    
    # Print summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\nInput Directory: {input_dir}")
    print(f"Total Files Analyzed: {total_files}")
    if error_count > 0:
        print(f"Files with Errors: {error_count}")
    
    print("\n" + "-" * 70)
    print("FILE LINE COUNT STATISTICS")
    print("-" * 70)
    print(f"  Total Lines of Code: {total_lines:,}")
    print(f"  Minimum: {line_stats['min']}")
    print(f"  Maximum: {line_stats['max']}")
    print(f"  Average: {line_stats['avg']:.2f}")
    
    print("\n" + "-" * 70)
    print("FUNCTION COUNT STATISTICS")
    print("-" * 70)
    print(f"  Total Functions: {total_functions:,}")
    print(f"  Files with Functions: {files_with_functions} ({files_with_functions/total_files*100:.1f}%)")
    print(f"  Minimum per file: {func_stats['min']}")
    print(f"  Maximum per file: {func_stats['max']}")
    print(f"  Average per file: {func_stats['avg']:.2f}")
    
    print("\n" + "-" * 70)
    print("PARAMETER COUNT STATISTICS")
    print("-" * 70)
    print(f"  Total Parameters: {total_parameters:,}")
    if total_functions > 0:
        print(f"  Average Parameters per Function: {total_parameters/total_functions:.2f}")
    print(f"  Minimum per file: {param_stats['min']}")
    print(f"  Maximum per file: {param_stats['max']}")
    print(f"  Average per file: {param_stats['avg']:.2f}")
    
    print("\n" + "-" * 70)
    print("RETURN ANNOTATION COUNT STATISTICS")
    print("-" * 70)
    print(f"  Total Return Annotations: {total_returns:,}")
    print(f"  Files with Return Annotations: {files_with_returns} ({files_with_returns/total_files*100:.1f}%)")
    if total_functions > 0:
        print(f"  Return Annotation Coverage: {total_returns/total_functions*100:.1f}%")
    print(f"  Minimum per file: {return_stats['min']}")
    print(f"  Maximum per file: {return_stats['max']}")
    print(f"  Average per file: {return_stats['avg']:.2f}")
    
    print("\n" + "-" * 70)
    print("CLASS COUNT STATISTICS")
    print("-" * 70)
    print(f"  Total Classes: {total_classes:,}")
    print(f"  Files with Classes: {files_with_classes} ({files_with_classes/total_files*100:.1f}%)")
    print(f"  Minimum per file: {class_stats['min']}")
    print(f"  Maximum per file: {class_stats['max']}")
    print(f"  Average per file: {class_stats['avg']:.2f}")
    
    print("\n" + "-" * 70)
    print("PARAMETERS WITH TYPE ANNOTATIONS STATISTICS")
    print("-" * 70)
    print(f"  Total: {total_param_annotated:,}")
    print(f"  Minimum per file: {param_annotated_stats['min']}")
    print(f"  Maximum per file: {param_annotated_stats['max']}")
    print(f"  Average per file: {param_annotated_stats['avg']:.2f}")
    
    print("\n" + "-" * 70)
    print("RETURN TYPES WITH TYPE ANNOTATIONS STATISTICS")
    print("-" * 70)
    print(f"  Total: {total_return_annotated:,}")
    print(f"  Minimum per file: {return_annotated_stats['min']}")
    print(f"  Maximum per file: {return_annotated_stats['max']}")
    print(f"  Average per file: {return_annotated_stats['avg']:.2f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    input_directory = "original_files"
    analyze_benchmarks(input_directory)

