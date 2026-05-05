"""
Find files that exist in BOTH inferred approaches but NOT in the original 500 untyped files
"""

from pathlib import Path
from collections import defaultdict


untyped_base = Path("./ManyTypes4py_benchmarks/500_untyped_files")
gpt5_1_infer_base = Path("./ManyTypes4py_benchmarks/gpt5_1_infer_stub_run/merged")
deepseek_3_base = Path("./ManyTypes4py_benchmarks/deepseek_3_stub_run/merged")

# Get all files
untyped_files = {f.name for f in untyped_base.glob("*.py")}
gpt5_1_files = {f.name for f in gpt5_1_infer_base.glob("*.py")}
deepseek_3_files = {f.name for f in deepseek_3_base.glob("*.py")}

print("=" * 100)
print("EXTRA FILES IN INFERRED APPROACHES")
print("=" * 100)

print(f"\nFile counts:")
print(f"  Original (500_untyped_files):        {len(untyped_files):3d} files")
print(f"  GPT-4 Inferred (gpt5_1_infer):      {len(gpt5_1_files):3d} files")
print(f"  DeepSeek (deepseek_3_stub):         {len(deepseek_3_files):3d} files")

# Find extra files
extra_in_gpt5_1 = gpt5_1_files - untyped_files
extra_in_deepseek = deepseek_3_files - untyped_files

# Files in BOTH inferred approaches but NOT in original
in_both_inferred = gpt5_1_files & deepseek_3_files
extra_in_both = in_both_inferred - untyped_files

print(f"\n" + "=" * 100)
print("FILES NOT IN ORIGINAL 500")
print("=" * 100)

print(f"\nExtra files in GPT-4 Inferred (not in 500):     {len(extra_in_gpt5_1):3d} files")
if extra_in_gpt5_1:
    print("  Files:")
    for fname in sorted(extra_in_gpt5_1):
        print(f"    - {fname}")

print(f"\nExtra files in DeepSeek (not in 500):           {len(extra_in_deepseek):3d} files")
if extra_in_deepseek:
    print("  Files:")
    for fname in sorted(extra_in_deepseek):
        print(f"    - {fname}")

print(f"\n" + "=" * 100)
print("FILES IN BOTH INFERRED APPROACHES BUT NOT IN ORIGINAL")
print("=" * 100)

print(f"\nFiles in BOTH inferred versions (not in 500):   {len(extra_in_both):3d} files")
if extra_in_both:
    print("\n  These files exist in BOTH gpt5_1_infer AND deepseek_3_stub but NOT in original:")
    for fname in sorted(extra_in_both):
        print(f"    - {fname}")
    
    # Check if these are duplicates or new files
    print(f"\n  Analysis:")
    print(f"    Total files in both inferred: {len(in_both_inferred)}")
    print(f"    Files in both that are in original: {len(in_both_inferred & untyped_files)}")
    print(f"    Files in both that are NOT in original: {len(extra_in_both)}")
else:
    print("\n  ✅ NO files exist in both inferred approaches that are missing from original")

# Venn analysis
print(f"\n" + "=" * 100)
print("DETAILED VENN ANALYSIS")
print("=" * 100)

# Files only in GPT-4 inferred
only_gpt5_1 = gpt5_1_files - deepseek_3_files - untyped_files
# Files only in DeepSeek
only_deepseek = deepseek_3_files - gpt5_1_files - untyped_files
# Files in both inferred but not original
both_inferred_not_orig = (gpt5_1_files & deepseek_3_files) - untyped_files
# Files in both inferred and original
both_inferred_and_orig = (gpt5_1_files & deepseek_3_files) & untyped_files

print(f"\nExtra files ONLY in GPT-4 Inferred (not in DeepSeek or original): {len(only_gpt5_1)}")
if only_gpt5_1:
    for fname in sorted(only_gpt5_1):
        print(f"  - {fname}")

print(f"\nExtra files ONLY in DeepSeek (not in GPT-4 Inferred or original): {len(only_deepseek)}")
if only_deepseek:
    for fname in sorted(only_deepseek):
        print(f"  - {fname}")

print(f"\nFiles in BOTH inferred versions but NOT in original: {len(both_inferred_not_orig)}")
if both_inferred_not_orig:
    for fname in sorted(both_inferred_not_orig):
        print(f"  - {fname}")
else:
    print("  ✅ No files")

print(f"\nFiles in BOTH inferred versions AND in original: {len(both_inferred_and_orig)}")

# Summary
print(f"\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

total_extra_gpt5_1 = len(extra_in_gpt5_1)
total_extra_deepseek = len(extra_in_deepseek)
shared_extra = len(extra_in_both)

print(f"""
Extra files analysis:
- GPT-4 Inferred has {total_extra_gpt5_1} files not in original
- DeepSeek has {total_extra_deepseek} files not in original
- BOTH inferred have {shared_extra} files not in original

This suggests:
- GPT-4 inferred generated {total_extra_gpt5_1} files that don't correspond to originals
- DeepSeek generated {total_extra_deepseek} files that don't correspond to originals
- {shared_extra} files appear in both inferred versions despite not having originals

Possible explanations:
1. Generated files (new synthetic examples)
2. Duplicate/variant versions
3. Post-processing artifacts
4. Different file naming/encoding
""")

# Check file sizes to understand if these are significant
if extra_in_both:
    print(f"\nChecking content of files in BOTH inferred but not original:")
    for fname in sorted(extra_in_both):
        gpt5_1_file = gpt5_1_infer_base / fname
        deepseek_file = deepseek_3_base / fname
        
        if gpt5_1_file.exists() and deepseek_file.exists():
            size_gpt5_1 = gpt5_1_file.stat().st_size
            size_deepseek = deepseek_file.stat().st_size
            
            print(f"\n  {fname}:")
            print(f"    GPT-4 size: {size_gpt5_1:,} bytes")
            print(f"    DeepSeek size: {size_deepseek:,} bytes")
            
            # Check if they're identical
            try:
                content_gpt5_1 = gpt5_1_file.read_text(encoding='utf-8', errors='ignore')
                content_deepseek = deepseek_file.read_text(encoding='utf-8', errors='ignore')
                
                if content_gpt5_1 == content_deepseek:
                    print(f"    ✅ Identical content")
                else:
                    print(f"    ❌ Different content")
                    # Show first few lines of each
                    lines_gpt5_1 = content_gpt5_1.split('\n')[:3]
                    lines_deepseek = content_deepseek.split('\n')[:3]
                    print(f"    GPT-4 first lines: {lines_gpt5_1}")
                    print(f"    DeepSeek first lines: {lines_deepseek}")
            except Exception as e:
                print(f"    Error reading content: {e}")
