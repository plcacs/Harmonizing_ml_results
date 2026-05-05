"""
Analyze file coverage between different type annotation approaches
"""

from pathlib import Path
from collections import defaultdict


untyped_base = Path("./ManyTypes4py_benchmarks/500_untyped_files")
gpt5_1_infer_base = Path("./ManyTypes4py_benchmarks/gpt5_1_infer_stub_run/merged")
gpt5_4_base = Path("./ManyTypes4py_benchmarks/gpt5_4_run")

# Get all files
untyped_files = {f.name for f in untyped_base.glob("*.py")}
inferred_files = {f.name for f in gpt5_1_infer_base.glob("*.py")}

gpt5_4_files = set()
gpt5_4_per_folder = defaultdict(set)
for folder_num in range(1, 18):
    folder = gpt5_4_base / str(folder_num)
    if folder.exists():
        for f in folder.glob("*.py"):
            gpt5_4_files.add(f.name)
            gpt5_4_per_folder[folder_num].add(f.name)

print("=" * 100)
print("FILE COVERAGE ANALYSIS")
print("=" * 100)

print(f"\nFile counts:")
print(f"  Untyped (500_untyped_files):         {len(untyped_files):3d} files")
print(f"  Inferred (gpt5_1_infer_stub):       {len(inferred_files):3d} files")
print(f"  Strict (gpt5_4_run):                {len(gpt5_4_files):3d} files")

print(f"\n" + "=" * 100)
print("COVERAGE COMPARISON")
print("=" * 100)

# Coverage analysis
inferred_in_untyped = inferred_files & untyped_files
strict_in_untyped = gpt5_4_files & untyped_files

inferred_not_in_untyped = inferred_files - untyped_files
strict_not_in_untyped = gpt5_4_files - untyped_files

untyped_not_in_inferred = untyped_files - inferred_files
untyped_not_in_strict = untyped_files - gpt5_4_files

print(f"\nInferred coverage:")
print(f"  Files matching untyped:     {len(inferred_in_untyped):3d} ({100*len(inferred_in_untyped)/len(untyped_files):.1f}%)")
print(f"  Inferred not in untyped:    {len(inferred_not_in_untyped):3d} files")
if inferred_not_in_untyped:
    print(f"    Examples: {list(inferred_not_in_untyped)[:3]}")

print(f"\nStrict coverage:")
print(f"  Files matching untyped:     {len(strict_in_untyped):3d} ({100*len(strict_in_untyped)/len(untyped_files):.1f}%)")
print(f"  Strict not in untyped:      {len(strict_not_in_untyped):3d} files")
if strict_not_in_untyped:
    print(f"    Examples: {list(strict_not_in_untyped)[:3]}")

print(f"\nUntyped NOT covered:")
print(f"  Missing from inferred:      {len(untyped_not_in_inferred):3d} ({100*len(untyped_not_in_inferred)/len(untyped_files):.1f}%)")
if untyped_not_in_inferred:
    print(f"    Examples: {list(untyped_not_in_inferred)[:3]}")

print(f"  Missing from strict:        {len(untyped_not_in_strict):3d} ({100*len(untyped_not_in_strict)/len(untyped_files):.1f}%)")
if untyped_not_in_strict:
    print(f"    Examples: {list(untyped_not_in_strict)[:3]}")

# Files in both approaches
both_approaches = inferred_in_untyped & strict_in_untyped
print(f"\nFiles in BOTH approaches:   {len(both_approaches):3d} ({100*len(both_approaches)/len(untyped_files):.1f}%)")

# Strict distribution
print(f"\n" + "=" * 100)
print("STRICT (gpt5_4_run) DISTRIBUTION BY FOLDER")
print("=" * 100)

for folder_num in range(1, 18):
    count = len(gpt5_4_per_folder[folder_num])
    print(f"  Folder {folder_num:2d}: {count:2d} files")

print(f"  Total:  {len(gpt5_4_files):3d} files")

# Summary
print(f"\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"""
Analysis Overview:
- Untyped source has 500 files
- Inferred approach covers {len(inferred_in_untyped)} files ({100*len(inferred_in_untyped)/len(untyped_files):.1f}%)
- Strict approach covers {len(strict_in_untyped)} files ({100*len(strict_in_untyped)/len(untyped_files):.1f}%)
- Both approaches cover {len(both_approaches)} files ({100*len(both_approaches)/len(untyped_files):.1f}%)

Key differences:
- {len(inferred_not_in_untyped)} extra files in inferred version
- {len(strict_not_in_untyped)} extra files in strict version
- {len(untyped_not_in_inferred)} untyped files not inferred
- {len(untyped_not_in_strict)} untyped files not strictly typed
""")
