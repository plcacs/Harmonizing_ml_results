"""
Enhanced AST Analysis with Async Functions and Control Flow Tracking
Provides detailed analysis of async functions, decorators, and control flow patterns.
"""

import ast
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class EnhancedASTAnalyzer:
    """Enhanced AST analysis with async function and control flow tracking."""
    
    def analyze_file(self, code: str, filename: str) -> Dict:
        """Analyze a Python file for async functions and control flow."""
        result = {
            "filename": filename,
            "has_async_functions": False,
            "async_count": 0,
            "async_functions": [],
            "decorators_count": 0,
            "control_flow_structures": {},
            "complexity_score": 0.0,
            "file_type": self._detect_file_type(filename)
        }
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            result["parse_error"] = True
            return result
        
        # Analyze async functions
        async_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                result["has_async_functions"] = True
                result["async_count"] += 1
                async_funcs.append(node.name)
            elif isinstance(node, ast.AsyncWith):
                result["async_count"] += 1
            elif isinstance(node, ast.AsyncFor):
                result["async_count"] += 1
        
        result["async_functions"] = async_funcs
        
        # Count decorators
        decorator_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorator_count += len(node.decorator_list)
        result["decorators_count"] = decorator_count
        
        # Analyze control flow
        control_flow = self._analyze_control_flow(tree)
        result["control_flow_structures"] = control_flow
        
        # Calculate complexity score
        result["complexity_score"] = self._calculate_complexity(tree)
        
        return result
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type based on filename."""
        if filename.startswith('test_'):
            return 'test'
        elif filename.startswith('conftest_'):
            return 'conftest'
        else:
            return 'source'
    
    def _analyze_control_flow(self, tree) -> Dict:
        """Analyze control flow structures in the file."""
        structures = {
            "if_statements": 0,
            "for_loops": 0,
            "while_loops": 0,
            "try_except": 0,
            "with_statements": 0,
            "nested_depth": 0
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
        
        # Calculate max nesting depth
        structures["nested_depth"] = self._calculate_nesting_depth(tree)
        
        return structures
    
    def _calculate_nesting_depth(self, node, depth=0, max_depth=0) -> int:
        """Calculate maximum nesting depth in the AST."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, 
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                new_depth = depth + 1
                max_depth = max(max_depth, new_depth)
                max_depth = max(max_depth, self._calculate_nesting_depth(child, new_depth, max_depth))
        return max_depth
    
    def _calculate_complexity(self, tree) -> float:
        """Calculate cyclomatic complexity score."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, (ast.BoolOp,)):
                complexity += len(node.values) - 1
        
        return complexity


def generate_async_analysis_report(compare_dir: str = "./original_vs_strict_comparison",
                                   original_dir: str = "./ManyTypes4py_benchmarks/original_files",
                                   output_dir: str = "./original_vs_strict_comparison"):
    """Generate detailed async function and control flow analysis report."""
    
    analyzer = EnhancedASTAnalyzer()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load original files
    original_base = Path(original_dir)
    original_files = {f.stem: f for f in original_base.glob("*.py")}
    
    print("Analyzing files for async functions and control flow...")
    
    # Analysis results
    async_stats = {
        'files_with_async': 0,
        'total_async_functions': 0,
        'test_files_with_async': 0,
        'async_test_functions': 0
    }
    
    control_flow_stats = defaultdict(int)
    complexity_distribution = defaultdict(int)
    file_details = []
    
    # Analyze original files
    for orig_file in original_files.values():
        try:
            code = orig_file.read_text(encoding='utf-8', errors='ignore')
            analysis = analyzer.analyze_file(code, orig_file.name)
            
            if 'parse_error' not in analysis:
                # Track async functions
                if analysis['has_async_functions']:
                    async_stats['files_with_async'] += 1
                    async_stats['total_async_functions'] += analysis['async_count']
                    
                    if analysis['file_type'] == 'test':
                        async_stats['test_files_with_async'] += 1
                        async_stats['async_test_functions'] += analysis['async_count']
                
                # Track control flow
                for control_type, count in analysis['control_flow_structures'].items():
                    if control_type != 'nested_depth':
                        control_flow_stats[control_type] += count
                
                # Track complexity
                complexity_bin = int(analysis['complexity_score'] / 10)
                complexity_distribution[complexity_bin] += 1
                
                # Store details for top files
                file_details.append({
                    'filename': analysis['filename'],
                    'async_count': analysis['async_count'],
                    'async_functions': ', '.join(analysis['async_functions'][:3]),
                    'complexity': f"{analysis['complexity_score']:.1f}",
                    'file_type': analysis['file_type'],
                    'nested_depth': analysis['control_flow_structures']['nested_depth']
                })
        
        except Exception as e:
            print(f"  Error analyzing {orig_file.name}: {e}")
    
    # Write detailed analysis to CSV
    analysis_file = output_path / "async_and_control_flow_analysis.csv"
    with open(analysis_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'async_count', 'async_functions', 
                                               'complexity', 'file_type', 'nested_depth'])
        writer.writeheader()
        for detail in sorted(file_details, key=lambda x: -int(float(x['complexity']))):
            writer.writerow(detail)
    
    # Print console summary
    print("\n" + "="*70)
    print("ASYNC FUNCTIONS & CONTROL FLOW ANALYSIS REPORT")
    print("="*70)
    
    print(f"\nASYNC FUNCTION STATISTICS:")
    print(f"  Files with async functions: {async_stats['files_with_async']}")
    print(f"  Total async constructs: {async_stats['total_async_functions']}")
    print(f"  Test files with async: {async_stats['test_files_with_async']}")
    print(f"  Async test constructs: {async_stats['async_test_functions']}")
    
    print(f"\nCONTROL FLOW STATISTICS (across all files):")
    print(f"  If statements: {control_flow_stats['if_statements']}")
    print(f"  For loops: {control_flow_stats['for_loops']}")
    print(f"  While loops: {control_flow_stats['while_loops']}")
    print(f"  Try-except blocks: {control_flow_stats['try_except']}")
    print(f"  With statements: {control_flow_stats['with_statements']}")
    
    print(f"\nCOMPLEXITY DISTRIBUTION (cyclomatic complexity):")
    total_files = len(file_details)
    for bin_num in sorted(complexity_distribution.keys()):
        count = complexity_distribution[bin_num]
        range_str = f"{bin_num*10}-{(bin_num+1)*10}"
        pct = count / total_files * 100 if total_files > 0 else 0
        print(f"  {range_str:>8}: {count:>3} files ({pct:>5.1f}%)")
    
    # Find highest complexity files
    print(f"\nTOP 10 HIGHEST COMPLEXITY FILES:")
    for i, detail in enumerate(sorted(file_details, key=lambda x: -float(x['complexity']))[:10], 1):
        async_info = f" [async: {detail['async_count']}]" if int(detail['async_count']) > 0 else ""
        print(f"  {i:2d}. {detail['filename']:40s} complexity={detail['complexity']:>5s}{async_info}")
    
    print("\n" + "="*70)
    print(f"Report saved to: {analysis_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_async_analysis_report()
    print("Async and control flow analysis complete!")
