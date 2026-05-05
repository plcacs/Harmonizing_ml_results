"""
Detailed analysis of actual structural changes in the 8 files
that differ between untyped and gpt5_4_run versions
"""

import ast
from pathlib import Path


def get_all_functions(tree):
    """Get all function names and their signatures."""
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            funcs[node.name] = args
    return funcs


def get_all_classes(tree):
    """Get all class names."""
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def count_structures(tree):
    """Count control flow structures."""
    counts = {
        "if": 0, "for": 0, "while": 0, "try": 0, "with": 0,
        "functions": 0, "classes": 0
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            counts["if"] += 1
        elif isinstance(node, ast.For):
            counts["for"] += 1
        elif isinstance(node, ast.While):
            counts["while"] += 1
        elif isinstance(node, ast.Try):
            counts["try"] += 1
        elif isinstance(node, ast.With):
            counts["with"] += 1
        elif isinstance(node, ast.FunctionDef):
            counts["functions"] += 1
        elif isinstance(node, ast.ClassDef):
            counts["classes"] += 1
    
    return counts


# Files that showed changes
changed_files = [
    ("awsclient_860c67.py", 2),
    ("manifest_47c52e.py", 6),
    ("test_payments_dde789.py", 6),
    ("test_ipython_b46143.py", 11),
    ("test_stack_unstack_b54311.py", 11),
    ("forms_6e49d8.py", 15),
    ("test_init_75a7d5.py", 15),
    ("test_sql_f4958a.py", 16),
]

untyped_base = Path("./ManyTypes4py_benchmarks/500_untyped_files")
typed_base = Path("./ManyTypes4py_benchmarks/gpt5_4_run")

print("=" * 100)
print("DETAILED ANALYSIS OF STRUCTURAL CHANGES")
print("=" * 100)

for filename, folder_num in changed_files:
    untyped_file = untyped_base / filename
    typed_file = typed_base / str(folder_num) / filename
    
    print(f"\n{'='*100}")
    print(f"FILE: {filename} (Folder {folder_num})")
    print(f"{'='*100}")
    
    if not untyped_file.exists():
        print(f"❌ Untyped file not found: {untyped_file}")
        continue
    
    if not typed_file.exists():
        print(f"❌ Typed file not found: {typed_file}")
        continue
    
    try:
        untyped_code = untyped_file.read_text(encoding='utf-8', errors='ignore')
        typed_code = typed_file.read_text(encoding='utf-8', errors='ignore')
        
        ast_untyped = ast.parse(untyped_code)
        ast_typed = ast.parse(typed_code)
        
        # Get file sizes
        untyped_lines = untyped_code.count('\n')
        typed_lines = typed_code.count('\n')
        
        print(f"\n📊 FILE METRICS:")
        print(f"  Untyped: {len(untyped_code):,} bytes, {untyped_lines:,} lines")
        print(f"  Typed:   {len(typed_code):,} bytes, {typed_lines:,} lines")
        print(f"  Diff:    {len(typed_code) - len(untyped_code):+,} bytes, {typed_lines - untyped_lines:+,} lines")
        
        # Get functions
        funcs_untyped = get_all_functions(ast_untyped)
        funcs_typed = get_all_functions(ast_typed)
        
        print(f"\n📋 FUNCTION ANALYSIS:")
        print(f"  Untyped: {len(funcs_untyped)} functions")
        print(f"  Typed:   {len(funcs_typed)} functions")
        
        # Find differences
        removed_funcs = set(funcs_untyped.keys()) - set(funcs_typed.keys())
        added_funcs = set(funcs_typed.keys()) - set(funcs_untyped.keys())
        
        if removed_funcs:
            print(f"  Removed: {list(removed_funcs)}")
        if added_funcs:
            print(f"  Added:   {list(added_funcs)}")
        
        # Check parameter changes
        for fname in set(funcs_untyped.keys()) & set(funcs_typed.keys()):
            if funcs_untyped[fname] != funcs_typed[fname]:
                print(f"  ✏️ {fname}:")
                print(f"     Untyped params: {funcs_untyped[fname]}")
                print(f"     Typed params:   {funcs_typed[fname]}")
        
        # Get classes
        classes_untyped = get_all_classes(ast_untyped)
        classes_typed = get_all_classes(ast_typed)
        
        print(f"\n🏛️ CLASS ANALYSIS:")
        print(f"  Untyped: {len(classes_untyped)} classes")
        print(f"  Typed:   {len(classes_typed)} classes")
        
        if classes_untyped != classes_typed:
            removed_classes = classes_untyped - classes_typed
            added_classes = classes_typed - classes_untyped
            if removed_classes:
                print(f"  Removed: {list(removed_classes)}")
            if added_classes:
                print(f"  Added:   {list(added_classes)}")
        
        # Count structures
        counts_untyped = count_structures(ast_untyped)
        counts_typed = count_structures(ast_typed)
        
        print(f"\n🔄 CONTROL FLOW STRUCTURES:")
        for key in ["if", "for", "while", "try", "with"]:
            diff = counts_typed[key] - counts_untyped[key]
            sign = "+" if diff > 0 else ""
            print(f"  {key:6s}: {counts_untyped[key]:3d} → {counts_typed[key]:3d} ({sign}{diff})")
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")
print("\nThese 8 files showed structural differences between untyped and typed versions.")
print("Most differences appear to be parameter/signature changes during type annotation.")
print("Control flow changes may be due to:")
print("  - Type checking guards (new if statements)")
print("  - Refactored code logic for typing")
print("  - Use of typing constructs (Generic, Protocol, etc.)")
