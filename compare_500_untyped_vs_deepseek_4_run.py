r"""
CORRECT AST Comparison: Match files by NAME
- Iterate over ManyTypes4py_benchmarks\deepseek_4_run files
- Find matching file by name in ManyTypes4py_benchmarks\500_untyped_files
- Compare them
"""

import ast
import csv
import json
import difflib
from pathlib import Path
from typing import Dict, List, Tuple


class StructuralASTComparator:
    """Compare AST structures between untyped and typed (strict) versions."""
    
    def compare_files(self, untyped_code: str, typed_code: str, filename: str) -> Dict:
        """Compare untyped vs typed files by AST structure."""
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
            ast_typed = ast.parse(typed_code)
        except SyntaxError as e:
            result["structural_changes_found"] = f"parse_error_typed: {str(e)}"
            return result
        
        # Compare classes
        classes_match, class_diffs = self._compare_classes(ast_untyped, ast_typed)
        if not classes_match:
            result["classes_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(class_diffs)
        
        # Compare methods
        methods_match, method_diffs = self._compare_methods(ast_untyped, ast_typed)
        if not methods_match:
            result["methods_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(method_diffs)
        
        # Compare function parameters
        params_match, param_diffs = self._compare_parameters(ast_untyped, ast_typed)
        if not params_match:
            result["parameters_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(param_diffs)
        
        # Compare control flow
        control_match, control_diffs = self._compare_control_flow(ast_untyped, ast_typed)
        if not control_match:
            result["control_flow_match"] = "no"
            result["total_differences"] += 1
            result["details"].extend(control_diffs)
        
        if result["total_differences"] > 0:
            result["structural_changes_found"] = ", ".join([d.get("type", "unknown") for d in result["details"]])
        
        return result
    
    def _compare_classes(self, ast_untyped, ast_typed) -> Tuple[bool, List]:
        """Compare class definitions."""
        classes_untyped = {node.name: node for node in ast.walk(ast_untyped) if isinstance(node, ast.ClassDef)}
        classes_typed = {node.name: node for node in ast.walk(ast_typed) if isinstance(node, ast.ClassDef)}
        
        diffs = []
        untyped_names = set(classes_untyped.keys())
        typed_names = set(classes_typed.keys())
        
        if untyped_names != typed_names:
            missing_in_typed = untyped_names - typed_names
            missing_in_untyped = typed_names - untyped_names
            if missing_in_typed:
                diffs.append({
                    "type": "class_missing_in_typed",
                    "classes": list(missing_in_typed),
                    "details": [{"name": cls, "reason": "removed in typed version"} for cls in missing_in_typed]
                })
            if missing_in_untyped:
                diffs.append({
                    "type": "class_added_in_typed",
                    "classes": list(missing_in_untyped),
                    "details": [{"name": cls, "reason": "added in typed version"} for cls in missing_in_untyped]
                })
            return False, diffs
        
        return True, []
    
    def _compare_methods(self, ast_untyped, ast_typed) -> Tuple[bool, List]:
        """Compare methods within classes."""
        diffs = []
        
        classes_untyped = {node.name: node for node in ast.walk(ast_untyped) if isinstance(node, ast.ClassDef)}
        classes_typed = {node.name: node for node in ast.walk(ast_typed) if isinstance(node, ast.ClassDef)}
        
        for class_name in classes_untyped:
            if class_name not in classes_typed:
                continue
            
            methods_untyped = {n.name: n for n in classes_untyped[class_name].body if isinstance(n, ast.FunctionDef)}
            methods_typed = {n.name: n for n in classes_typed[class_name].body if isinstance(n, ast.FunctionDef)}
            
            untyped_names = set(methods_untyped.keys())
            typed_names = set(methods_typed.keys())
            
            if untyped_names != typed_names:
                missing_in_typed = untyped_names - typed_names
                missing_in_untyped = typed_names - untyped_names
                
                if missing_in_typed:
                    diffs.append({
                        "type": "method_removed_in_typed",
                        "class": class_name,
                        "methods": list(missing_in_typed),
                        "details": [{"name": m, "class": class_name} for m in missing_in_typed]
                    })
                if missing_in_untyped:
                    diffs.append({
                        "type": "method_added_in_typed",
                        "class": class_name,
                        "methods": list(missing_in_untyped),
                        "details": [{"name": m, "class": class_name} for m in missing_in_untyped]
                    })
                return False, diffs
        
        return True, []
    
    def _compare_module_functions(self, ast_untyped, ast_typed) -> Tuple[bool, List]:
        """Compare module-level functions (excluding class methods)."""
        diffs = []
        
        # Get all module-level function definitions (direct children of Module)
        funcs_untyped = {node.name: node for node in ast_untyped.body if isinstance(node, ast.FunctionDef)}
        funcs_typed = {node.name: node for node in ast_typed.body if isinstance(node, ast.FunctionDef)}
        
        untyped_names = set(funcs_untyped.keys())
        typed_names = set(funcs_typed.keys())
        
        if untyped_names != typed_names:
            missing_in_typed = untyped_names - typed_names
            missing_in_untyped = typed_names - untyped_names
            
            if missing_in_typed:
                diffs.append({
                    "type": "function_removed_in_typed",
                    "functions": list(missing_in_typed),
                    "details": [{"name": f} for f in missing_in_typed]
                })
            if missing_in_untyped:
                diffs.append({
                    "type": "function_added_in_typed",
                    "functions": list(missing_in_untyped),
                    "details": [{"name": f} for f in missing_in_untyped]
                })
            return False, diffs
        
        return True, []
    
    def _compare_parameters(self, ast_untyped, ast_typed) -> Tuple[bool, List]:
        """Compare function signatures."""
        diffs = []
        
        funcs_untyped = {node.name: node for node in ast.walk(ast_untyped) if isinstance(node, ast.FunctionDef)}
        funcs_typed = {node.name: node for node in ast.walk(ast_typed) if isinstance(node, ast.FunctionDef)}
        
        for func_name in funcs_untyped:
            if func_name not in funcs_typed:
                continue
            
            params_untyped = self._get_function_params(funcs_untyped[func_name])
            params_typed = self._get_function_params(funcs_typed[func_name])
            
            if params_untyped != params_typed:
                diffs.append({
                    "type": "parameter_mismatch",
                    "function": func_name,
                    "untyped_params": params_untyped,
                    "typed_params": params_typed,
                    "details": {
                        "function": func_name,
                        "untyped_count": len(params_untyped),
                        "typed_count": len(params_typed),
                        "untyped_signature": ", ".join(params_untyped),
                        "typed_signature": ", ".join(params_typed)
                    }
                })
                return False, diffs
        
        return True, []
    
    def _compare_control_flow(self, ast_untyped, ast_typed) -> Tuple[bool, List]:
        """Compare control flow structures."""
        diffs = []
        
        control_untyped = self._count_control_structures(ast_untyped)
        control_typed = self._count_control_structures(ast_typed)
        
        if control_untyped != control_typed:
            differences = {}
            for key in control_untyped:
                if control_untyped[key] != control_typed[key]:
                    differences[key] = {
                        "untyped": control_untyped[key],
                        "typed": control_typed[key],
                        "delta": control_typed[key] - control_untyped[key]
                    }
            
            diffs.append({
                "type": "control_flow_mismatch",
                "untyped": control_untyped,
                "typed": control_typed,
                "differences": differences
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


def compare_by_filename(output_dir: str = "./comparison_500_untyped_vs_deepseek_4_run"):
    """
    Compare files by matching filenames:
    - Iterate over deepseek_4_run files
    - Find matching file in 500_untyped_files by name
    - Compare them
    """
    
    untyped_base = Path("./ManyTypes4py_benchmarks/500_untyped_files")
    typed_base = Path("./ManyTypes4py_benchmarks/deepseek_4_run")
    
    # Check if paths exist
    if not untyped_base.exists():
        print(f"Error: {untyped_base} not found")
        return
    if not typed_base.exists():
        print(f"Error: {typed_base} not found")
        return
    
    # Build index of untyped files by name
    untyped_by_name = {}
    for untyped_file in untyped_base.glob("*.py"):
        untyped_by_name[untyped_file.name] = untyped_file
    
    print(f"Indexed {len(untyped_by_name)} untyped files\n")
    
    comparator = StructuralASTComparator()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Track results
    all_results = []
    files_with_changes = []
    parse_error_files = []  # Track files with parse errors
    
    # Iterate over all folders in deepseek_4_run
    for folder_num in range(1, 18):
        typed_folder = typed_base / str(folder_num)
        
        if not typed_folder.exists():
            print(f"Skipping folder {folder_num} - not found")
            continue
        
        typed_files = sorted(typed_folder.glob("*.py"))
        print(f"Folder {folder_num}: {len(typed_files)} typed files", end=" → ")
        
        folder_results = []
        matched_count = 0
        
        # For each typed file, find matching untyped file by name
        for typed_file in typed_files:
            filename = typed_file.name
            
            # Look for matching untyped file
            if filename not in untyped_by_name:
                print(f"\nWarning: No matching untyped file for {filename}")
                continue
            
            untyped_file = untyped_by_name[filename]
            matched_count += 1
            
            try:
                untyped_code = untyped_file.read_text(encoding='utf-8', errors='ignore')
                typed_code = typed_file.read_text(encoding='utf-8', errors='ignore')
                
                # Check if both files can be parsed
                untyped_parse_error = None
                typed_parse_error = None
                
                try:
                    ast.parse(untyped_code)
                except SyntaxError as e:
                    untyped_parse_error = str(e)
                
                try:
                    ast.parse(typed_code)
                except SyntaxError as e:
                    typed_parse_error = str(e)
                
                # If either file has parse error, track it
                if untyped_parse_error or typed_parse_error:
                    parse_error_files.append({
                        "filename": filename,
                        "folder": folder_num,
                        "untyped_error": untyped_parse_error,
                        "typed_error": typed_parse_error
                    })
                    # Still count as matched but don't do full comparison
                    all_results.append({
                        "filename": filename,
                        "folder": folder_num,
                        "total_differences": 0,
                        "parse_error": True,
                        "untyped_error": untyped_parse_error,
                        "typed_error": typed_parse_error
                    })
                    continue
                
                # Only do full comparison if both parse successfully
                comparison = comparator.compare_files(untyped_code, typed_code, filename)
                comparison["folder"] = folder_num
                comparison["untyped_path"] = str(untyped_file.relative_to(untyped_base))
                comparison["typed_path"] = str(typed_file.relative_to(typed_base))
                comparison["untyped_code"] = untyped_code
                comparison["typed_code"] = typed_code
                
                folder_results.append(comparison)
                all_results.append(comparison)
                
                # Track files with changes
                if comparison["total_differences"] > 0:
                    files_with_changes.append(comparison)
                    
            except Exception as e:
                print(f"\nError comparing {filename}: {e}")
        
        print(f"{matched_count} matched")
        
        # Write CSV for this folder
        if folder_results:
            output_file = Path(output_dir) / f"comparison_results_{folder_num}.csv"
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "filename",
                    "folder",
                    "classes_match",
                    "methods_match",
                    "parameters_match",
                    "control_flow_match",
                    "total_differences",
                    "structural_changes_found"
                ])
                
                for result in folder_results:
                    writer.writerow([
                        result.get("filename", ""),
                        result.get("folder", ""),
                        result.get("classes_match", ""),
                        result.get("methods_match", ""),
                        result.get("parameters_match", ""),
                        result.get("control_flow_match", ""),
                        result.get("total_differences", 0),
                        result.get("structural_changes_found", "")
                    ])
    
    # Write aggregate summary
    print(f"\n{'='*80}")
    print(f"Total files compared: {len(all_results)}")
    print(f"Files with structural changes: {len(files_with_changes)}")
    
    if all_results:
        pct = (len(files_with_changes) / len(all_results)) * 100
        print(f"Percentage changed: {pct:.2f}%")
    
    # Write files with changes
    if files_with_changes:
        output_file = Path(output_dir) / "files_with_changes.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename",
                "folder",
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
                    result.get("folder", ""),
                    result.get("classes_match", ""),
                    result.get("methods_match", ""),
                    result.get("parameters_match", ""),
                    result.get("control_flow_match", ""),
                    result.get("total_differences", 0),
                    result.get("structural_changes_found", "")
                ])
    
    # Helper function to extract detailed change information
    def extract_detailed_changes(details):
        """Extract detailed information about all changes."""
        changes = []
        
        for detail in details:
            detail_type = detail.get("type", "")
            
            if detail_type == "class_missing_in_typed":
                for item in detail.get("details", []):
                    changes.append({
                        "type": "class_removed",
                        "name": item.get("name"),
                        "reason": item.get("reason", "removed in typed version")
                    })
            
            elif detail_type == "class_added_in_typed":
                for item in detail.get("details", []):
                    changes.append({
                        "type": "class_added",
                        "name": item.get("name"),
                        "reason": item.get("reason", "added in typed version")
                    })
            
            elif detail_type == "method_removed_in_typed":
                for item in detail.get("details", []):
                    changes.append({
                        "type": "method_removed",
                        "class": item.get("class"),
                        "name": item.get("name")
                    })
            
            elif detail_type == "method_added_in_typed":
                for item in detail.get("details", []):
                    changes.append({
                        "type": "method_added",
                        "class": item.get("class"),
                        "name": item.get("name")
                    })
            
            elif detail_type == "function_removed_in_typed":
                for item in detail.get("details", []):
                    changes.append({
                        "type": "method_removed",
                        "name": item.get("name"),
                        "class": ""
                    })
            
            elif detail_type == "function_added_in_typed":
                for item in detail.get("details", []):
                    changes.append({
                        "type": "method_added",
                        "name": item.get("name"),
                        "class": ""
                    })
            
            elif detail_type == "parameter_mismatch":
                details_obj = detail.get("details", {})
                changes.append({
                    "type": "parameter_mismatch",
                    "function": details_obj.get("function"),
                    "untyped_params": details_obj.get("untyped_signature", ""),
                    "typed_params": details_obj.get("typed_signature", ""),
                    "untyped_count": details_obj.get("untyped_count"),
                    "typed_count": details_obj.get("typed_count")
                })
            
            elif detail_type == "control_flow_mismatch":
                diffs = detail.get("differences", {})
                changes.append({
                    "type": "control_flow_change",
                    "untyped_counts": detail.get("untyped", {}),
                    "typed_counts": detail.get("typed", {}),
                    "changes": diffs
                })
        
        return changes
    
    # Write summary JSON
    summary = {
        "total_files_compared": len(all_results),
        "files_with_changes": len(files_with_changes),
        "percentage_changed": (len(files_with_changes) / len(all_results) * 100) if all_results else 0,
        "changed_files": [
            {
                "filename": r["filename"],
                "folder": r["folder"],
                "total_differences": r["total_differences"],
                "structural_changes": r["structural_changes_found"],
                "detailed_changes": extract_detailed_changes(r.get("details", []))
            }
            for r in files_with_changes
        ]
    }
    
    summary_file = Path(output_dir) / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate simplified markdown report
    def generate_markdown_report(summary, output_dir, files_with_changes_list, parse_errors=None):
        """Generate a simplified markdown report organized by folder."""
        if parse_errors is None:
            parse_errors = []
        
        md = []
        md.append("# AST Comparison Report: deepseek_4_run vs 500_untyped_files\n\n")
        md.append(f"**Total files compared**: {summary['total_files_compared']}\n")
        md.append(f"**Files with changes**: {summary['files_with_changes']}\n")
        md.append(f"**Files with parse errors**: {len(parse_errors)}\n")
        md.append(f"**Percentage changed**: {summary['percentage_changed']:.2f}%\n\n")
        md.append("---\n\n")
        
        # Show parse errors first if any
        if parse_errors:
            md.append("## ⚠️ Files with Parse Errors (Cannot Generate AST)\n\n")
            parse_by_folder = {}
            for pe in parse_errors:
                folder = pe['folder']
                if folder not in parse_by_folder:
                    parse_by_folder[folder] = []
                parse_by_folder[folder].append(pe)
            
            for folder_num in sorted(parse_by_folder.keys()):
                md.append(f"### Folder {folder_num}\n\n")
                for pe in parse_by_folder[folder_num]:
                    md.append(f"- **{pe['filename']}**\n")
                    if pe['untyped_error']:
                        md.append(f"  - Untyped error: {pe['untyped_error'][:80]}...\n")
                    if pe['typed_error']:
                        md.append(f"  - Typed error: {pe['typed_error'][:80]}...\n")
                md.append("\n")
            
            md.append("---\n\n")
        
        # Group files by folder using summary data
        files_by_folder = {}
        for file_data in summary['changed_files']:
            folder = file_data['folder']
            if folder not in files_by_folder:
                files_by_folder[folder] = []
            files_by_folder[folder].append(file_data)
        
        # Process each folder
        for folder_num in sorted(files_by_folder.keys()):
            folder_files = files_by_folder[folder_num]
            md.append(f"## Folder {folder_num}\n\n")
            
            # Process each file in folder
            for file_data in folder_files:
                md.append(f"### {file_data['filename']}\n\n")
                
                changes = file_data['detailed_changes']
                has_class_changes = False
                has_param_changes = False
                
                # Classes removed
                removed_classes = [c for c in changes if c['type'] == 'class_removed']
                if removed_classes:
                    has_class_changes = True
                    md.append("**Classes Removed:**\n")
                    for change in removed_classes:
                        md.append(f"- `{change['name']}`\n")
                    md.append("\n")
                
                # Classes added
                added_classes = [c for c in changes if c['type'] == 'class_added']
                if added_classes:
                    has_class_changes = True
                    md.append("**Classes Added:**\n")
                    for change in added_classes:
                        md.append(f"- `{change['name']}`\n")
                    md.append("\n")
                
                # Methods/Functions removed
                removed_methods = [c for c in changes if c['type'] == 'method_removed']
                if removed_methods:
                    md.append("**Methods/Functions Removed:**\n")
                    for change in removed_methods:
                        class_name = change.get('class', '')
                        func_name = change['name']
                        if class_name:
                            md.append(f"- `{class_name}.{func_name}`\n")
                        else:
                            md.append(f"- `{func_name}`\n")
                    md.append("\n")
                
                # Methods/Functions added
                added_methods = [c for c in changes if c['type'] == 'method_added']
                if added_methods:
                    md.append("**Methods/Functions Added:**\n")
                    for change in added_methods:
                        class_name = change.get('class', '')
                        func_name = change['name']
                        if class_name:
                            md.append(f"- `{class_name}.{func_name}`\n")
                        else:
                            md.append(f"- `{func_name}`\n")
                    md.append("\n")
                
                # Parameter mismatches
                param_changes = [c for c in changes if c['type'] == 'parameter_mismatch']
                if param_changes:
                    has_param_changes = True
                    md.append("**Parameter Changes:**\n\n")
                    for change in param_changes:
                        md.append(f"**Function**: `{change['function']}`\n\n")
                        md.append("| | Untyped | Typed |\n")
                        md.append("|---|---|---|\n")
                        md.append(f"| **Params** | `{change['untyped_params']}` | `{change['typed_params']}` |\n\n")
                
                if not has_class_changes and not has_param_changes and not removed_methods and not added_methods:
                    md.append("*(Other structural changes - no class, method, or parameter modifications)*\n\n")
            
            md.append("---\n\n")
        
        return "".join(md)
    
    md_content = generate_markdown_report(summary, output_dir, files_with_changes, parse_error_files)
    md_file = Path(output_dir) / "CHANGES_REPORT.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\nAll results saved to {output_dir}/")
    print(f"  - summary.json: Detailed JSON with all changes")
    print(f"  - CHANGES_REPORT.md: Human-readable markdown report")
    print(f"  - files_with_changes.csv: CSV summary")


if __name__ == "__main__":
    print("Starting comparison: deepseek_4_run vs 500_untyped_files\n")
    compare_by_filename()
    print("\nDone!")
