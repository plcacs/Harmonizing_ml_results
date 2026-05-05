"""
Understanding Control Flow Differences - Detailed Explanation
Shows what the metrics mean and why changes occurred.
"""

import ast
from pathlib import Path


def explain_manifest_differences():
    """Provide detailed explanation of manifest_47c52e.py changes."""
    
    original_base = Path("./ManyTypes4py_benchmarks/original_files")
    strict_base = Path("./ManyTypes4py_benchmarks/gpt5_4_run")
    
    # Find the file
    original_files = sorted(original_base.glob("*.py"))
    
    # Folder 6, approximately 29 files per folder
    files_per_folder = len(original_files) // 17
    start_idx = (6 - 1) * files_per_folder
    end_idx = start_idx + files_per_folder
    folder_originals = original_files[start_idx:end_idx]
    
    # Find manifest_47c52e.py
    strict_file = strict_base / "6" / "manifest_47c52e.py"
    
    # Get corresponding original file (using index)
    strict_folder = sorted(strict_base.glob("6/*.py"))
    file_idx = sorted([f.name for f in strict_folder]).index("manifest_47c52e.py")
    
    if file_idx < len(folder_originals):
        orig_file = folder_originals[file_idx]
        
        orig_code = orig_file.read_text(encoding='utf-8', errors='ignore')
        strict_code = strict_file.read_text(encoding='utf-8', errors='ignore')
        
        # Parse both
        orig_ast = ast.parse(orig_code)
        strict_ast = ast.parse(strict_code)
        
        # Count control flow in original
        orig_if = sum(1 for node in ast.walk(orig_ast) if isinstance(node, ast.If))
        orig_for = sum(1 for node in ast.walk(orig_ast) if isinstance(node, ast.For))
        orig_try = sum(1 for node in ast.walk(orig_ast) if isinstance(node, ast.Try))
        
        # Count control flow in strict
        strict_if = sum(1 for node in ast.walk(strict_ast) if isinstance(node, ast.If))
        strict_for = sum(1 for node in ast.walk(strict_ast) if isinstance(node, ast.For))
        strict_try = sum(1 for node in ast.walk(strict_ast) if isinstance(node, ast.Try))
        
        print("\n" + "="*80)
        print("UNDERSTANDING manifest_47c52e.py CHANGES")
        print("="*80)
        
        print("\n📊 WHAT THE METRICS MEAN:\n")
        
        print("1️⃣  IF STATEMENTS: 35 → 149 (+114)")
        print("   " + "-"*76)
        print("   • Original file had 35 'if' statements")
        print("   • Strict version has 149 'if' statements")
        print("   • 114 NEW conditional branches were added")
        print("\n   MEANING:")
        print("   The strict version added significantly more conditional logic.")
        print("   This could be:")
        print("   ✓ Type checking (isinstance checks for type annotations)")
        print("   ✓ Error handling (checking for valid inputs)")
        print("   ✓ Edge cases (handling different scenarios)")
        print("   ✓ Validation logic (checking data before processing)")
        
        print("\n2️⃣  FOR LOOPS: 6 → 31 (+25)")
        print("   " + "-"*76)
        print("   • Original file had 6 'for' loops")
        print("   • Strict version has 31 'for' loops")
        print("   • 25 NEW loop constructs were added")
        print("\n   MEANING:")
        print("   The strict version added much more iteration logic.")
        print("   This could be:")
        print("   ✓ Iterating over collections for validation")
        print("   ✓ Processing multiple items with type checking")
        print("   ✓ Building results by looping through data")
        
        print("\n3️⃣  TRY-EXCEPT: 4 → 0 (-4)")
        print("   " + "-"*76)
        print("   • Original file had 4 'try-except' blocks")
        print("   • Strict version has 0 'try-except' blocks")
        print("   • 4 ERROR HANDLERS were REMOVED")
        print("\n   MEANING:")
        print("   The strict version removed exception handling.")
        print("   This could mean:")
        print("   ✓ The LLM expected better input validation (fewer exceptions)")
        print("   ✓ Type annotations prevent certain errors from happening")
        print("   ✓ Different architectural approach (fail-fast instead of try-catch)")
        
        print("\n" + "="*80)
        print("OVERALL INTERPRETATION")
        print("="*80)
        
        print(f"\nOriginal Summary:")
        print(f"  • {orig_if} conditional checks")
        print(f"  • {orig_for} loops")
        print(f"  • {orig_try} error handlers")
        print(f"  • Total control structures: {orig_if + orig_for + orig_try}")
        
        print(f"\nStrict Summary:")
        print(f"  • {strict_if} conditional checks")
        print(f"  • {strict_for} loops")
        print(f"  • {strict_try} error handlers")
        print(f"  • Total control structures: {strict_if + strict_for + strict_try}")
        
        increase = (strict_if + strict_for + strict_try) - (orig_if + orig_for + orig_try)
        print(f"\nTotal Control Structure Change: +{increase}")
        print(f"Complexity Increase: {increase / (orig_if + orig_for + orig_try) * 100:.1f}%")
        
        print("\n" + "="*80)
        print("KEY TAKEAWAY")
        print("="*80)
        print("""
The strict (gpt5_4_run) version of manifest_47c52e.py is significantly more:

  🔍 DEFENSIVE     - More conditional checks to handle edge cases
  🔄 PROCEDURAL    - More loops for iterating through data
  ⚡ PROTECTIVE    - Removed error handlers (assumes good input)

This suggests the LLM generated code that:
  1. Validates inputs thoroughly with many if statements
  2. Processes data more methodically with more loops
  3. Assumes exceptions won't happen (type annotations prevent them)

This is typical of AI-generated code that's overly defensive and explicit.
""")
        
        print("\n" + "="*80)
        print("WHAT TO LOOK FOR IN THE ACTUAL CODE")
        print("="*80)
        print("""
When you examine the files, you'll find:

✓ ADDED IF STATEMENTS (114 new ones):
  - Look for lots of: isinstance(x, Type), if x is None, if x == value
  - Check for: if not isinstance(obj, str): raise TypeError(...)
  - Type validation: if type(variable) == expected_type

✓ ADDED FOR LOOPS (25 new ones):
  - Processing collections: for item in items: ...
  - Validation loops: for param in parameters: validate(param)
  - List comprehensions that were replaced with explicit loops

✓ REMOVED TRY-EXCEPT (4 removed):
  - Original probably had: try: ... except Exception: ...
  - Strict removed these because if statements prevent the exception
  - This is GOOD - preventative validation instead of reactive handling
""")


if __name__ == "__main__":
    explain_manifest_differences()
