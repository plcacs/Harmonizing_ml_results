"""
Detailed Difference Analyzer
Shows exactly what changed between original and strict versions for files with mismatches.
"""

import ast
import csv
from pathlib import Path
from typing import Dict, List, Tuple


class DetailedDifferenceAnalyzer:
    """Analyze and report detailed differences in function signatures, methods, and control flow."""
    
    def compare_files_detailed(self, original_code: str, strict_code: str, filename: str) -> Dict:
        """Compare files and provide detailed difference information."""
        result = {
            "filename": filename,
            "parameter_differences": [],
            "method_differences": [],
            "control_flow_differences": [],
            "class_differences": []
        }
        
        try:
            ast_original = ast.parse(original_code)
            ast_strict = ast.parse(strict_code)
        except SyntaxError:
            result["parse_error"] = True
            return result
        
        # Compare function parameters
        param_diffs = self._compare_function_parameters(ast_original, ast_strict)
        if param_diffs:
            result["parameter_differences"] = param_diffs
        
        # Compare methods
        method_diffs = self._compare_class_methods(ast_original, ast_strict)
        if method_diffs:
            result["method_differences"] = method_diffs
        
        # Compare control flow
        control_diffs = self._compare_control_flow_detailed(ast_original, ast_strict)
        if control_diffs:
            result["control_flow_differences"] = control_diffs
        
        return result
    
    def _compare_function_parameters(self, ast_original, ast_strict) -> List[Dict]:
        """Find specific parameter differences in functions."""
        diffs = []
        
        # Get all functions from both files
        funcs_original = {node.name: node for node in ast.walk(ast_original) 
                         if isinstance(node, ast.FunctionDef)}
        funcs_strict = {node.name: node for node in ast.walk(ast_strict) 
                       if isinstance(node, ast.FunctionDef)}
        
        # Compare parameters for matching function names
        for func_name in funcs_original:
            if func_name not in funcs_strict:
                continue
            
            orig_params = self._extract_params_detailed(funcs_original[func_name])
            strict_params = self._extract_params_detailed(funcs_strict[func_name])
            
            if orig_params != strict_params:
                diffs.append({
                    "function": func_name,
                    "original": orig_params,
                    "strict": strict_params,
                    "added": [p for p in strict_params if p not in orig_params],
                    "removed": [p for p in orig_params if p not in strict_params]
                })
        
        return diffs
    
    def _extract_params_detailed(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract detailed parameter info including annotations."""
        params = []
        
        for arg in func_node.args.args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)
        
        for arg in func_node.args.posonlyargs:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)
        
        for arg in func_node.args.kwonlyargs:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)
        
        if func_node.args.vararg:
            param = f"*{func_node.args.vararg.arg}"
            if func_node.args.vararg.annotation:
                param += f": {ast.unparse(func_node.args.vararg.annotation)}"
            params.append(param)
        
        if func_node.args.kwarg:
            param = f"**{func_node.args.kwarg.arg}"
            if func_node.args.kwarg.annotation:
                param += f": {ast.unparse(func_node.args.kwarg.annotation)}"
            params.append(param)
        
        return params
    
    def _compare_class_methods(self, ast_original, ast_strict) -> List[Dict]:
        """Find specific method differences in classes."""
        diffs = []
        
        classes_original = {node.name: node for node in ast.walk(ast_original) 
                           if isinstance(node, ast.ClassDef)}
        classes_strict = {node.name: node for node in ast.walk(ast_strict) 
                         if isinstance(node, ast.ClassDef)}
        
        for class_name in classes_original:
            if class_name not in classes_strict:
                continue
            
            methods_original = {n.name: n for n in classes_original[class_name].body 
                              if isinstance(n, ast.FunctionDef)}
            methods_strict = {n.name: n for n in classes_strict[class_name].body 
                            if isinstance(n, ast.FunctionDef)}
            
            # Check for removed/added methods
            removed = set(methods_original.keys()) - set(methods_strict.keys())
            added = set(methods_strict.keys()) - set(methods_original.keys())
            
            if removed or added:
                diffs.append({
                    "class": class_name,
                    "removed_methods": list(removed),
                    "added_methods": list(added)
                })
        
        return diffs
    
    def _compare_control_flow_detailed(self, ast_original, ast_strict) -> List[Dict]:
        """Find specific control flow differences."""
        diffs = []
        
        original_counts = self._count_control_structures(ast_original)
        strict_counts = self._count_control_structures(ast_strict)
        
        changes = {}
        for structure_type in original_counts:
            if original_counts[structure_type] != strict_counts[structure_type]:
                changes[structure_type] = {
                    "original": original_counts[structure_type],
                    "strict": strict_counts[structure_type],
                    "change": strict_counts[structure_type] - original_counts[structure_type]
                }
        
        if changes:
            diffs.append({"control_flow_changes": changes})
        
        return diffs
    
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


def analyze_changed_files(compare_dir: str = "./original_vs_strict_comparison",
                         original_dir: str = "./ManyTypes4py_benchmarks/original_files",
                         strict_base: str = "./ManyTypes4py_benchmarks/gpt5_4_run",
                         output_file: str = "./original_vs_strict_comparison/detailed_differences.txt"):
    """Analyze files with changes and produce detailed report."""
    
    analyzer = DetailedDifferenceAnalyzer()
    output_path = Path(output_file)
    
    # Read files with changes
    changes_csv = Path(compare_dir) / "files_with_changes.csv"
    
    if not changes_csv.exists():
        print(f"Error: {changes_csv} not found")
        return
    
    files_to_analyze = []
    with open(changes_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            files_to_analyze.append({
                'filename': row['filename'],
                'folder': row['folder'],
                'change_type': row['changes']
            })
    
    print(f"Analyzing {len(files_to_analyze)} files with changes...\n")
    
    # Analyze each file
    detailed_report = []
    
    for file_info in files_to_analyze:
        filename = file_info['filename']
        folder = file_info['folder']
        
        # Find original file
        original_base = Path(original_dir)
        original_files = list(original_base.glob("*.py"))
        all_original = sorted(original_files)
        
        # Distribute files: approximately 500 / 17 = ~29 files per folder
        files_per_folder = len(all_original) // 17
        start_idx = (int(folder) - 1) * files_per_folder
        end_idx = start_idx + files_per_folder if int(folder) < 17 else len(all_original)
        
        folder_originals = all_original[start_idx:end_idx]
        strict_folder = Path(strict_base) / folder
        strict_files = sorted([f for f in strict_folder.glob("*.py")])
        
        # Match by index
        file_idx = sorted([f.name for f in strict_files]).index(filename)
        
        if file_idx < len(folder_originals):
            orig_file = folder_originals[file_idx]
            strict_file = Path(strict_base) / folder / filename
            
            if strict_file.exists():
                try:
                    orig_code = orig_file.read_text(encoding='utf-8', errors='ignore')
                    strict_code = strict_file.read_text(encoding='utf-8', errors='ignore')
                    
                    analysis = analyzer.compare_files_detailed(orig_code, strict_code, filename)
                    
                    if 'parse_error' not in analysis:
                        detailed_report.append({
                            'filename': filename,
                            'folder': folder,
                            'change_type': file_info['change_type'],
                            'analysis': analysis
                        })
                        
                        print(f"✓ Analyzed: {filename} (Folder {folder})")
                
                except Exception as e:
                    print(f"✗ Error analyzing {filename}: {e}")
    
    # Write detailed report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED DIFFERENCE ANALYSIS\n")
        f.write("Original Files vs Strict (gpt5_4_run) Versions\n")
        f.write("="*80 + "\n\n")
        
        for item in detailed_report:
            f.write(f"\nFILE: {item['filename']} (Folder {item['folder']})\n")
            f.write(f"Change Type: {item['change_type']}\n")
            f.write("-"*80 + "\n")
            
            analysis = item['analysis']
            
            # Parameter differences
            if analysis['parameter_differences']:
                f.write("\n📋 PARAMETER DIFFERENCES:\n")
                for param_diff in analysis['parameter_differences']:
                    f.write(f"\n  Function: {param_diff['function']}\n")
                    f.write(f"  Original:  {param_diff['original']}\n")
                    f.write(f"  Strict:    {param_diff['strict']}\n")
                    if param_diff['added']:
                        f.write(f"  Added:     {param_diff['added']}\n")
                    if param_diff['removed']:
                        f.write(f"  Removed:   {param_diff['removed']}\n")
            
            # Method differences
            if analysis['method_differences']:
                f.write("\n📦 METHOD DIFFERENCES:\n")
                for method_diff in analysis['method_differences']:
                    f.write(f"\n  Class: {method_diff['class']}\n")
                    if method_diff['removed_methods']:
                        f.write(f"  Removed: {method_diff['removed_methods']}\n")
                    if method_diff['added_methods']:
                        f.write(f"  Added:   {method_diff['added_methods']}\n")
            
            # Control flow differences
            if analysis['control_flow_differences']:
                f.write("\n🔄 CONTROL FLOW DIFFERENCES:\n")
                for cf_diff in analysis['control_flow_differences']:
                    for struct_type, changes in cf_diff['control_flow_changes'].items():
                        f.write(f"  {struct_type}: {changes['original']} → {changes['strict']} ({changes['change']:+d})\n")
            
            f.write("\n" + "="*80 + "\n")
    
    # Print console summary
    print(f"\n✅ Detailed report saved to: {output_path}\n")
    
    with open(output_path, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    analyze_changed_files()
