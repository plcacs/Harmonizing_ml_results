"""
AST Comparison: 500 Untyped Files vs gpt5_1_infer_stub_run/merged
Match files by NAME and compare structural changes (ignoring type annotations)
"""

import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


class StructuralASTComparator:
    """Compare AST structures between untyped and typed (inferred stub) versions."""
    
    def compare_files(self, untyped_code: str, inferred_code: str, filename: str) -> Dict:
        """Compare untyped vs inferred files by AST structure."""
        result = {
            "filename": filename,
            "classes_match": "yes",
            "methods_match": "yes",
            "parameters_match": "yes",
            "control_flow_match": "yes",
            "total_differences": 0,
            "structural_changes_found": "no",
            "details": []
        }
        
        try:
            ast_untyped = ast.parse(untyped_code)
        except SyntaxError as e:
            result["structural_changes_found"] = f"parse_error_untyped: {str(e)}"
            return result
        
        try:
            ast_inferred = ast.parse(inferred_code)
        except SyntaxError as e:
            result["structural_changes_found"] = f"parse_error_inferred: {str(e)}"
            return result
        
        # Compare classes
        classes_match, class_diffs = self._compare_classes(ast_untyped, ast_inferred)
        if not classes_match:
            result["classes_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(class_diffs)
        
        # Compare methods
        methods_match, method_diffs = self._compare_methods(ast_untyped, ast_inferred)
        if not methods_match:
            result["methods_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(method_diffs)
        
        # Compare function parameters (structural, not type annotations)
        params_match, param_diffs = self._compare_parameters(ast_untyped, ast_inferred)
        if not params_match:
            result["parameters_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(param_diffs)
        
        # Compare control flow
        control_match, control_diffs = self._compare_control_flow(ast_untyped, ast_inferred)
        if not control_match:
            result["control_flow_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(control_diffs)
        
        if result["total_differences"] > 0:
            result["structural_changes_found"] = ", ".join([d.get("type", "unknown") for d in result["details"]])
        
        return result
    
    def _compare_classes(self, ast_untyped, ast_inferred) -> Tuple[bool, List]:
        """Compare class definitions."""
        classes_untyped = {node.name for node in ast.walk(ast_untyped) if isinstance(node, ast.ClassDef)}
        classes_inferred = {node.name for node in ast.walk(ast_inferred) if isinstance(node, ast.ClassDef)}
        
        diffs = []
        if classes_untyped != classes_inferred:
            missing_in_inferred = classes_untyped - classes_inferred
            missing_in_untyped = classes_inferred - classes_untyped
            if missing_in_inferred:
                diffs.append({"type": "class_missing_in_inferred", "classes": list(missing_in_inferred)})
            if missing_in_untyped:
                diffs.append({"type": "class_added_in_inferred", "classes": list(missing_in_untyped)})
            return False, diffs
        
        return True, []
    
    def _compare_methods(self, ast_untyped, ast_inferred) -> Tuple[bool, List]:
        """Compare methods within classes."""
        diffs = []
        
        classes_untyped = {node.name: node for node in ast.walk(ast_untyped) if isinstance(node, ast.ClassDef)}
        classes_inferred = {node.name: node for node in ast.walk(ast_inferred) if isinstance(node, ast.ClassDef)}
        
        for class_name in classes_untyped:
            if class_name not in classes_inferred:
                continue
            
            methods_untyped = {n.name for n in classes_untyped[class_name].body if isinstance(n, ast.FunctionDef)}
            methods_inferred = {n.name for n in classes_inferred[class_name].body if isinstance(n, ast.FunctionDef)}
            
            if methods_untyped != methods_inferred:
                missing_in_inferred = methods_untyped - methods_inferred
                missing_in_untyped = methods_inferred - methods_untyped
                if missing_in_inferred:
                    diffs.append({
                        "type": "method_removed_in_inferred",
                        "class": class_name,
                        "methods": list(missing_in_inferred)
                    })
                if missing_in_untyped:
                    diffs.append({
                        "type": "method_added_in_inferred",
                        "class": class_name,
                        "methods": list(missing_in_untyped)
                    })
                return False, diffs
        
        return True, []
    
    def _compare_parameters(self, ast_untyped, ast_inferred) -> Tuple[bool, List]:
        """Compare function signatures (parameter count/names, not type annotations)."""
        diffs = []
        
        funcs_untyped = {node.name: node for node in ast.walk(ast_untyped) if isinstance(node, ast.FunctionDef)}
        funcs_inferred = {node.name: node for node in ast.walk(ast_inferred) if isinstance(node, ast.FunctionDef)}
        
        for func_name in funcs_untyped:
            if func_name not in funcs_inferred:
                continue
            
            params_untyped = self._get_function_params(funcs_untyped[func_name])
            params_inferred = self._get_function_params(funcs_inferred[func_name])
            
            if params_untyped != params_inferred:
                diffs.append({
                    "type": "parameter_mismatch",
                    "function": func_name,
                    "untyped_params": len(params_untyped),
                    "inferred_params": len(params_inferred)
                })
                return False, diffs
        
        return True, []
    
    def _compare_control_flow(self, ast_untyped, ast_inferred) -> Tuple[bool, List]:
        """Compare control flow structures."""
        diffs = []
        
        control_untyped = self._count_control_structures(ast_untyped)
        control_inferred = self._count_control_structures(ast_inferred)
        
        if control_untyped != control_inferred:
            diffs.append({
                "type": "control_flow_mismatch",
                "untyped": control_untyped,
                "inferred": control_inferred
            })
            return False, diffs
        
        return True, []
    
    def _get_function_params(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract function parameter names (ignore type annotations)."""
        params = []
        for arg in func_node.args.args:
            params.append(arg.arg)
        for arg in func_node.args.posonlyargs:
            params.append(arg.arg)
        for arg in func_node.args.kwonlyargs:
            params.append(arg.arg)
        return params
    
    def _count_control_structures(self, tree) -> Dict:
        """Count control flow structures."""
        structures = {
            "if_statements": 0,
            "for_loops": 0,
            "while_loops": 0,
            "try_except": 0,
            "with_statements": 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                structures["if_statements"] += 1
            elif isinstance(node, ast.For):
                structures["for_loops"] += 1
            elif isinstance(node, ast.While):
                structures["while_loops"] += 1
            elif isinstance(node, ast.Try):
                structures["try_except"] += 1
            elif isinstance(node, ast.With):
                structures["with_statements"] += 1
        
        return structures


def compare_500_vs_gpt5_1_infer_stub(output_dir: str = "./comparison_500_untyped_vs_gpt5_1_infer_stub"):
    """
    Compare files by matching filenames:
    - Iterate over gpt5_1_infer_stub_run/merged files
    - Find matching file in 500_untyped_files by name
    - Compare them for structural changes (ignoring type annotations)
    """
    
    untyped_base = Path("./ManyTypes4py_benchmarks/500_untyped_files")
    inferred_base = Path("./ManyTypes4py_benchmarks/gpt5_1_infer_stub_run/merged")
    
    # Check if paths exist
    if not untyped_base.exists():
        print(f"Error: {untyped_base} not found")
        return
    if not inferred_base.exists():
        print(f"Error: {inferred_base} not found")
        return
    
    # Build index of untyped files by name
    untyped_by_name = {}
    for untyped_file in untyped_base.glob("*.py"):
        untyped_by_name[untyped_file.name] = untyped_file
    
    print(f"Indexed {len(untyped_by_name)} untyped files")
    
    # Get all inferred files
    inferred_files = sorted(inferred_base.glob("*.py"))
    print(f"Found {len(inferred_files)} inferred files\n")
    
    comparator = StructuralASTComparator()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Track results
    all_results = []
    files_with_changes = []
    matched_count = 0
    
    print(f"Comparing files...")
    print(f"{'='*80}")
    
    # For each inferred file, find matching untyped file by name
    for inferred_file in inferred_files:
        filename = inferred_file.name
        
        # Look for matching untyped file
        if filename not in untyped_by_name:
            continue
        
        untyped_file = untyped_by_name[filename]
        matched_count += 1
        
        try:
            untyped_code = untyped_file.read_text(encoding='utf-8', errors='ignore')
            inferred_code = inferred_file.read_text(encoding='utf-8', errors='ignore')
            
            comparison = comparator.compare_files(untyped_code, inferred_code, filename)
            comparison["inferred_path"] = str(inferred_file.relative_to(inferred_base))
            
            all_results.append(comparison)
            
            # Track files with changes
            if comparison["total_differences"] > 0:
                files_with_changes.append(comparison)
                print(f"✓ {filename} - {comparison['structural_changes_found']}")
                    
        except Exception as e:
            print(f"✗ Error comparing {filename}: {e}")
    
    # Write summary
    print(f"\n{'='*80}")
    print(f"Matched files: {matched_count}")
    print(f"Total files compared: {len(all_results)}")
    print(f"Files with structural changes: {len(files_with_changes)}")
    
    if all_results:
        pct = (len(files_with_changes) / len(all_results)) * 100
        print(f"Percentage changed: {pct:.2f}%")
    
    # Write CSV results
    output_file = Path(output_dir) / "comparison_results.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "classes_match",
            "methods_match",
            "parameters_match",
            "control_flow_match",
            "total_differences",
            "structural_changes_found"
        ])
        
        for result in all_results:
            writer.writerow([
                result.get("filename", ""),
                result.get("classes_match", ""),
                result.get("methods_match", ""),
                result.get("parameters_match", ""),
                result.get("control_flow_match", ""),
                result.get("total_differences", 0),
                result.get("structural_changes_found", "")
            ])
    
    # Write files with changes
    if files_with_changes:
        output_file = Path(output_dir) / "files_with_changes.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename",
                "classes_match",
                "methods_match",
                "parameters_match",
                "control_flow_match",
                "total_differences",
                "structural_changes_found"
            ])
            
            for result in files_with_changes:
                writer.writerow([
                    result.get("filename", ""),
                    result.get("classes_match", ""),
                    result.get("methods_match", ""),
                    result.get("parameters_match", ""),
                    result.get("control_flow_match", ""),
                    result.get("total_differences", 0),
                    result.get("structural_changes_found", "")
                ])
    
    # Write summary JSON
    summary = {
        "comparison_type": "500_untyped_files vs gpt5_1_infer_stub_run/merged",
        "total_files_compared": len(all_results),
        "files_with_changes": len(files_with_changes),
        "percentage_changed": (len(files_with_changes) / len(all_results) * 100) if all_results else 0,
        "changed_files": [
            {
                "filename": r["filename"],
                "total_differences": r["total_differences"],
                "structural_changes": r["structural_changes_found"]
            }
            for r in sorted(files_with_changes, key=lambda x: x["filename"])
        ]
    }
    
    summary_file = Path(output_dir) / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}/")
    print(f"  - comparison_results.csv (all files)")
    print(f"  - files_with_changes.csv (only changed files)")
    print(f"  - summary.json (aggregate statistics)")


if __name__ == "__main__":
    print("Starting comparison: 500 Untyped Files vs gpt5_1_infer_stub_run/merged\n")
    compare_500_vs_gpt5_1_infer_stub()
    print("\nDone!")
