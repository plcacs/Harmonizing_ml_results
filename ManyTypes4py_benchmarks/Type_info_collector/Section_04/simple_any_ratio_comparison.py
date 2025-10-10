#!/usr/bin/env python3
"""
Simple script to find files where untyped version has 20% more any_ratio than partial/fully typed versions.
"""

import json

# Load the three JSON files
with open('per_file_any_percentage/O3-mini_1st_run/per_file_any_percentage.json', 'r') as f:
    untyped = json.load(f)

with open('per_file_any_percentage_partially_typed_user_annotated/O3-mini_Partially_Typed/per_file_any_percentage.json', 'r') as f:
    partial = json.load(f)

with open('per_file_any_percentage_partially_typed_user_annotated/O3-mini_User_Annotated/per_file_any_percentage.json', 'r') as f:
    fully = json.load(f)

print("=== FILES WHERE UNTYPED HAS 20% MORE ANY RATIO ===")
print()

# Find files where untyped has 20% more than partial
print("1. UNTYPED vs PARTIAL (untyped has 20%+ more):")
count1 = 0
for filename in untyped:
    if filename in partial:
        untyped_ratio = untyped[filename]['any_percentage']
        partial_ratio = partial[filename]['any_percentage']
        if untyped_ratio - partial_ratio >= 20:
            print(f"   {filename}: {untyped_ratio:.1f}% vs {partial_ratio:.1f}% (diff: +{untyped_ratio-partial_ratio:.1f}%)")
            count1 += 1
print(f"   Total: {count1} files")
print()

# Find files where untyped has 20% more than fully
print("2. UNTYPED vs FULLY (untyped has 20%+ more):")
count2 = 0
for filename in untyped:
    if filename in fully:
        untyped_ratio = untyped[filename]['any_percentage']
        fully_ratio = fully[filename]['any_percentage']
        if untyped_ratio - fully_ratio >= 20:
            print(f"   {filename}: {untyped_ratio:.1f}% vs {fully_ratio:.1f}% (diff: +{untyped_ratio-fully_ratio:.1f}%)")
            count2 += 1
print(f"   Total: {count2} files")
print()

# Summary
print("=== SUMMARY ===")
print(f"Files where untyped > partial by 20%+: {count1}")
print(f"Files where untyped > fully by 20%+: {count2}")
print(f"Total files analyzed: {len(untyped)}")
