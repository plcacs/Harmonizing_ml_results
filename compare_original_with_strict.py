"""
AST Structural Comparison between original (untyped) files and strict (gpt5_4_run) versions.
Outputs CSV with structural differences for each folder (1-17).
"""

import ast
import csv
from pathlib import Path
from typing import Dict, List, Tuple


class StructuralASTComparator:
    """Compare AST structures between two Python files."""
    
    def __init__(self):
        self.differences = []
    
    def compare_files(self, original_code: str, strict_code: str, filename: str) -> Dict:
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
            ast_original = ast.parse(original_code)
        except SyntaxError as e:
            result["structural_changes_found"] = f"parse_error_original"
            return result
        
        try:
            ast_strict = ast.parse(strict_code)
        except SyntaxError as e:
            result["structural_changes_found"] = f"parse_error_strict"
            return result
        
        # Compare classes
        classes_match, class_diffs = self._compare_classes(ast_original, ast_strict)
        if not classes_match:
            result["classes_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(class_diffs)
        
        # Compare methods
        methods_match, method_diffs = self._compare_methods(ast_original, ast_strict)
        if not methods_match:
            result["methods_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(method_diffs)
        
        # Compare function parameters
        params_match, param_diffs = self._compare_parameters(ast_original, ast_strict)
        if not params_match:
            result["parameters_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(param_diffs)
        
        # Compare control flow
        control_match, control_diffs = self._compare_control_flow(ast_original, ast_strict)
        if not control_match:
            result["control_flow_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(control_diffs)
        
        # Set structural_changes_found
        if result["total_differences"] > 0:
            result["structural_changes_found"] = ", ".join([d.get("type", "unknown") for d in result["details"]])
        
        return result
    
    def _compare_classes(self, ast_original, ast_strict) -> Tuple[bool, List]:
        """Compare class definitions."""
        classes_original = {node.name for node in ast.walk(ast_original) if isinstance(node, ast.ClassDef)}
        classes_strict = {node.name for node in ast.walk(ast_strict) if isinstance(node, ast.ClassDef)}
        
        diffs = []
        if classes_original != classes_strict:
            missing_in_strict = classes_original - classes_strict
            missing_in_original = classes_strict - classes_original
            if missing_in_strict:
                diffs.append({"type": "class_missing_in_strict", "classes": list(missing_in_strict)})
            if missing_in_original:
                diffs.append({"type": "class_missing_in_original", "classes": list(missing_in_original)})
            return False, diffs
        
        return True, []
    
    def _compare_methods(self, ast_original, ast_strict) -> Tuple[bool, List]:
        """Compare methods within classes."""
        diffs = []
        
        classes_original = {node.name: node for node in ast.walk(ast_original) if isinstance(node, ast.ClassDef)}
        classes_strict = {node.name: node for node in ast.walk(ast_strict) if isinstance(node, ast.ClassDef)}
        
        for class_name in classes_original:
            if class_name not in classes_strict:
                continue
            
            methods_original = {n.name for n in classes_original[class_name].body if isinstance(n, ast.FunctionDef)}
            methods_strict = {n.name for n in classes_strict[class_name].body if isinstance(n, ast.FunctionDef)}
            
            if methods_original != methods_strict:
                missing_in_strict = methods_original - methods_strict
                missing_in_original = methods_strict - methods_original
                if missing_in_strict:
                    diffs.append({
                        "type": "method_missing_in_strict",
                        "class": class_name,
                        "methods": list(missing_in_strict)
                    })
                if missing_in_original:
                    diffs.append({
                        "type": "method_missing_in_original",
                        "class": class_name,
                        "methods": list(missing_in_original)
                    })
                return False, diffs
        
        return True, []
    
    def _compare_parameters(self, ast_original, ast_strict) -> Tuple[bool, List]:
        """Compare function signatures (parameters)."""
        diffs = []
        
        funcs_original = {node.name: node for node in ast.walk(ast_original) if isinstance(node, ast.FunctionDef)}
        funcs_strict = {node.name: node for node in ast.walk(ast_strict) if isinstance(node, ast.FunctionDef)}
        
        for func_name in funcs_original:
            if func_name not in funcs_strict:
                continue
            
            params_original = self._get_function_params(funcs_original[func_name])
            params_strict = self._get_function_params(funcs_strict[func_name])
            
            if params_original != params_strict:
                diffs.append({
                    "type": "parameter_mismatch",
                    "function": func_name,
                    "original_params": len(params_original),
                    "strict_params": len(params_strict),
                    "original_names": params_original,
                    "strict_names": params_strict
                })
                return False, diffs
        
        return True, []
    
    def _compare_control_flow(self, ast_original, ast_strict) -> Tuple[bool, List]:
        """Compare control flow structures (if, for, while, try)."""
        diffs = []
        
        control_original = self._count_control_structures(ast_original)
        control_strict = self._count_control_structures(ast_strict)
        
        if control_original != control_strict:
            diffs.append({
                "type": "control_flow_mismatch",
                "original": control_original,
                "strict": control_strict
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


def write_csv(results: List[Dict], folder_num: int, output_dir: str = "./original_vs_strict_comparison"):
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
        
        # Write rows
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


def compare_original_with_strict(output_dir: str = "./original_vs_strict_comparison"):
    """Compare original files with strict (gpt5_4_run) files."""
    
    # Use local paths
    original_base = Path("./ManyTypes4py_benchmarks/original_files")
    strict_base = Path("./ManyTypes4py_benchmarks/gpt5_4_run")
    
    # Check if paths exist
    if not original_base.exists():
        print(f"Error: {original_base} not found")
        return
    if not strict_base.exists():
        print(f"Error: {strict_base} not found")
        return
    
    comparator = StructuralASTComparator()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get all files from original_files
    original_files = list(original_base.glob("*.py"))
    print(f"Found {len(original_files)} original files")
    
    # Create a mapping of strict files to original files
    original_files_dict = {f.stem: f for f in original_files}
    
    # Compare all folders (1-17) with original files
    for folder_num in range(1, 18):
        s_folder = strict_base / str(folder_num)
        
        if not s_folder.exists():
            print(f"Skipping folder {folder_num} - not found")
            continue
        
        print(f"Comparing folder {folder_num}...", end=" ")
        folder_results = []
        matched_count = 0
        
        # Get all Python files from strict folder
        strict_files = sorted(s_folder.glob("*.py"))
        
        # For each strict file, try to find matching original file
        for strict_file in strict_files:
            orig_file = None
            
            # Try direct stem match first
            if strict_file.stem in original_files_dict:
                orig_file = original_files_dict[strict_file.stem]
            else:
                # If no direct match, use sequential matching from original files
                # Match file index to distribute original files across folders
                file_index = sorted([f.name for f in s_folder.glob("*.py")]).index(strict_file.name)
                all_original = sorted(original_files)
                
                # Distribute original files: approximately 500 / 17 = ~29 files per folder
                files_per_folder = len(all_original) // 17
                start_idx = (folder_num - 1) * files_per_folder
                end_idx = start_idx + files_per_folder if folder_num < 17 else len(all_original)
                
                folder_originals = all_original[start_idx:end_idx]
                if file_index < len(folder_originals):
                    orig_file = folder_originals[file_index]
            
            if orig_file:
                try:
                    strict_code = strict_file.read_text(encoding='utf-8', errors='ignore')
                    orig_code = orig_file.read_text(encoding='utf-8', errors='ignore')
                    
                    comparison = comparator.compare_files(orig_code, strict_code, strict_file.name)
                    folder_results.append(comparison)
                    matched_count += 1
                except Exception as e:
                    print(f"Error comparing {strict_file.name}: {e}")
        
        # Write CSV for this folder
        if folder_results:
            write_csv(folder_results, folder_num, output_dir)
            print(f"({matched_count} files) → {output_dir}/comparison_results_{folder_num}.csv")
        else:
            print(f"(0 files)")
    
    print(f"\nAll results saved to {output_dir}/")

if __name__ == "__main__":
    print("Starting original vs strict AST structural comparison...")
    compare_original_with_strict()
    print("Done!")
