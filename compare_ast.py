"""
AST Structural Comparison between non-strict (gpt5_2_run) and strict (gpt5_4_run) versions.
Outputs CSV with structural differences only.
"""

import ast
import csv
from pathlib import Path
from typing import Dict, List, Tuple


class StructuralASTComparator:
    """Compare AST structures between two Python files."""
    
    def __init__(self):
        self.differences = []
    
    def compare_files(self, non_strict_code: str, strict_code: str, filename: str) -> Dict:
        """Compare two Python files by their AST structure."""
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
        
        # Parse both files
        try:
            ast_ns = ast.parse(non_strict_code)
        except SyntaxError as e:
            result["structural_changes_found"] = f"parse_error_non_strict"
            return result
        
        try:
            ast_s = ast.parse(strict_code)
        except SyntaxError as e:
            result["structural_changes_found"] = f"parse_error_strict"
            return result
        
        # Compare classes
        classes_match, class_diffs = self._compare_classes(ast_ns, ast_s)
        if not classes_match:
            result["classes_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(class_diffs)
        
        # Compare methods
        methods_match, method_diffs = self._compare_methods(ast_ns, ast_s)
        if not methods_match:
            result["methods_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(method_diffs)
        
        # Compare function parameters
        params_match, param_diffs = self._compare_parameters(ast_ns, ast_s)
        if not params_match:
            result["parameters_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(param_diffs)
        
        # Compare control flow
        control_match, control_diffs = self._compare_control_flow(ast_ns, ast_s)
        if not control_match:
            result["control_flow_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(control_diffs)
        
        # Set structural_changes_found
        if result["total_differences"] > 0:
            result["structural_changes_found"] = ", ".join([d.get("type", "unknown") for d in result["details"]])
        
        return result
    
    def _compare_classes(self, ast_ns, ast_s) -> Tuple[bool, List]:
        """Compare class definitions."""
        classes_ns = {node.name for node in ast.walk(ast_ns) if isinstance(node, ast.ClassDef)}
        classes_s = {node.name for node in ast.walk(ast_s) if isinstance(node, ast.ClassDef)}
        
        diffs = []
        if classes_ns != classes_s:
            missing_in_strict = classes_ns - classes_s
            missing_in_ns = classes_s - classes_ns
            if missing_in_strict:
                diffs.append({"type": "class_missing_in_strict", "classes": list(missing_in_strict)})
            if missing_in_ns:
                diffs.append({"type": "class_missing_in_ns", "classes": list(missing_in_ns)})
            return False, diffs
        
        return True, []
    
    def _compare_methods(self, ast_ns, ast_s) -> Tuple[bool, List]:
        """Compare methods within classes."""
        diffs = []
        
        classes_ns = {node.name: node for node in ast.walk(ast_ns) if isinstance(node, ast.ClassDef)}
        classes_s = {node.name: node for node in ast.walk(ast_s) if isinstance(node, ast.ClassDef)}
        
        for class_name in classes_ns:
            if class_name not in classes_s:
                continue
            
            methods_ns = {n.name for n in classes_ns[class_name].body if isinstance(n, ast.FunctionDef)}
            methods_s = {n.name for n in classes_s[class_name].body if isinstance(n, ast.FunctionDef)}
            
            if methods_ns != methods_s:
                missing_in_strict = methods_ns - methods_s
                missing_in_ns = methods_s - methods_ns
                if missing_in_strict:
                    diffs.append({
                        "type": "method_missing_in_strict",
                        "class": class_name,
                        "methods": list(missing_in_strict)
                    })
                if missing_in_ns:
                    diffs.append({
                        "type": "method_missing_in_ns",
                        "class": class_name,
                        "methods": list(missing_in_ns)
                    })
                return False, diffs
        
        return True, []
    
    def _compare_parameters(self, ast_ns, ast_s) -> Tuple[bool, List]:
        """Compare function signatures (parameters)."""
        diffs = []
        
        funcs_ns = {node.name: node for node in ast.walk(ast_ns) if isinstance(node, ast.FunctionDef)}
        funcs_s = {node.name: node for node in ast.walk(ast_s) if isinstance(node, ast.FunctionDef)}
        
        for func_name in funcs_ns:
            if func_name not in funcs_s:
                continue
            
            params_ns = self._get_function_params(funcs_ns[func_name])
            params_s = self._get_function_params(funcs_s[func_name])
            
            if params_ns != params_s:
                diffs.append({
                    "type": "parameter_mismatch",
                    "function": func_name,
                    "non_strict_params": len(params_ns),
                    "strict_params": len(params_s),
                    "non_strict_names": params_ns,
                    "strict_names": params_s
                })
                return False, diffs
        
        return True, []
    
    def _compare_control_flow(self, ast_ns, ast_s) -> Tuple[bool, List]:
        """Compare control flow structures (if, for, while, try)."""
        diffs = []
        
        control_ns = self._count_control_structures(ast_ns)
        control_s = self._count_control_structures(ast_s)
        
        if control_ns != control_s:
            diffs.append({
                "type": "control_flow_mismatch",
                "non_strict": control_ns,
                "strict": control_s
            })
            return False, diffs
        
        return True, []
    
    def _get_function_params(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract function parameter names."""
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




def write_csv(results: List[Dict], folder_num: int, output_dir: str = "./ast_comparison"):
    """Write results to CSV file per folder."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as comparison_results_{folder_num}.csv
    output_file = Path(output_dir) / f"comparison_results_{folder_num}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "filename",
            "classes_match",
            "methods_match",
            "parameters_match",
            "control_flow_match",
            "total_differences",
            "structural_changes_found"
        ])
        
        # Write rows (no folder column needed, it's the folder name)
        for result in results:
            writer.writerow([
                result.get("filename", ""),
                result.get("classes_match", ""),
                result.get("methods_match", ""),
                result.get("parameters_match", ""),
                result.get("control_flow_match", ""),
                result.get("total_differences", 0),
                result.get("structural_changes_found", "")
            ])


def clone_and_compare(output_dir: str = "./ast_comparison"):
    """Compare files from local ManyTypes4py_benchmarks folders."""
    
    # Use local paths
    non_strict_base = Path("./ManyTypes4py_benchmarks/gpt5_2_run")
    strict_base = Path("./ManyTypes4py_benchmarks/gpt5_4_run")
    
    # Check if paths exist
    if not non_strict_base.exists():
        print(f"Error: {non_strict_base} not found")
        return
    if not strict_base.exists():
        print(f"Error: {strict_base} not found")
        return
    
    comparator = StructuralASTComparator()
    
    # Create ast_comparison directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Compare all folders (1-17)
    for folder_num in range(1, 18):
        ns_folder = non_strict_base / str(folder_num)
        s_folder = strict_base / str(folder_num)
        
        if not ns_folder.exists() or not s_folder.exists():
            print(f"Skipping folder {folder_num} - not found")
            continue
        
        print(f"Comparing folder {folder_num}...", end=" ")
        folder_results = []
        folder_count = 0
        
        # Get matching Python files from both folders
        for py_file in sorted(ns_folder.glob("*.py")):
            matching_strict = s_folder / py_file.name
            
            if matching_strict.exists():
                try:
                    ns_code = py_file.read_text(encoding='utf-8', errors='ignore')
                    s_code = matching_strict.read_text(encoding='utf-8', errors='ignore')
                    
                    comparison = comparator.compare_files(ns_code, s_code, py_file.name)
                    folder_results.append(comparison)
                    folder_count += 1
                except Exception as e:
                    print(f"Error comparing {py_file.name}: {e}")
        
        # Write CSV for this folder
        if folder_results:
            write_csv(folder_results, folder_num, output_dir)
            print(f"({folder_count} files) → {output_dir}/comparison_results_{folder_num}.csv")
        else:
            print(f"(0 files)")
    
    print(f"\nAll results saved to {output_dir}/")

if __name__ == "__main__":
    print("Starting AST structural comparison...")
    clone_and_compare()
    print("Done!")
