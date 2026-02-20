import json
import os
import shutil

BASE_DIR = os.path.dirname(__file__)
UNTYPED_PATH = os.path.join(
    BASE_DIR, "..", "mypy_results", "mypy_outputs",
    "mypy_results_untyped_with_errors.json"
)
ORIGINAL_PATH = os.path.join(
    BASE_DIR, "..", "mypy_results", "mypy_outputs",
    "mypy_results_original_files_with_errors.json"
)
ORIGINAL_FILES_DIR = os.path.join(BASE_DIR, "..", "original_files")
OUTPUT_PATH = os.path.join(BASE_DIR, "top_200_programs.json")
TOP_N = 200

# Step 1: Filter compiled files from untyped version
with open(UNTYPED_PATH, "r") as f:
    untyped_data = json.load(f)

compiled_names = {
    name for name, info in untyped_data.items()
    if info.get("isCompiled") is True
}
print(f"Total files (untyped): {len(untyped_data)}")
print(f"Compiled files (untyped): {len(compiled_names)}")

# Step 2: From compiled set, get annotation stats from original version
with open(ORIGINAL_PATH, "r") as f:
    original_data = json.load(f)

compiled = {
    name: original_data[name] for name in compiled_names
    if name in original_data
    and original_data[name]["stats"]["parameters_with_annotations"] > 0
}
print(f"With annotations (from original): {len(compiled)}")

# Count lines for each compiled file from original_files
for name, info in compiled.items():
    filepath = os.path.join(ORIGINAL_FILES_DIR, name)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            info["line_count"] = sum(1 for _ in f)
    else:
        info["line_count"] = 0

found = sum(1 for info in compiled.values() if info["line_count"] > 0)
print(f"Files with line count: {found}/{len(compiled)}")

# Sort by parameters_with_annotations (primary), total_parameters (secondary), line_count (tertiary)
ranked = sorted(
    compiled.items(),
    key=lambda x: (
        x[1]["stats"]["parameters_with_annotations"],
        x[1]["stats"]["total_parameters"],
        x[1]["line_count"],
    ),
    reverse=True
)

selected = ranked[:TOP_N]

result = {name: info for name, info in selected}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=4)

print(f"\nSelected {len(result)} files")
print(f"Annotated params range: {selected[-1][1]['stats']['parameters_with_annotations']} - {selected[0][1]['stats']['parameters_with_annotations']}")
print(f"Total params range: {selected[-1][1]['stats']['total_parameters']} - {selected[0][1]['stats']['total_parameters']}")
print(f"Line count range: {min(v['line_count'] for v in result.values())} - {max(v['line_count'] for v in result.values())}")
print(f"Saved to: {OUTPUT_PATH}")

# Copy selected files to Human_annotations_detailed_analysis/selected_files
COPY_DIR = os.path.join(BASE_DIR, "selected_files")
os.makedirs(COPY_DIR, exist_ok=True)

copied = 0
for name in result:
    src = os.path.join(ORIGINAL_FILES_DIR, name)
    dst = os.path.join(COPY_DIR, name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1

print(f"Copied {copied} files to: {COPY_DIR}")
