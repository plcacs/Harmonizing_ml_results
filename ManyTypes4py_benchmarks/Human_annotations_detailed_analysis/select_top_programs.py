import json
import os

BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(
    BASE_DIR, "..", "mypy_results", "mypy_outputs",
    "mypy_results_untyped_with_errors.json"
)
ORIGINAL_FILES_DIR = os.path.join(BASE_DIR, "..", "original_files")
OUTPUT_PATH = os.path.join(BASE_DIR, "top_200_programs.json")
TOP_N = 200

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

compiled = {
    name: info for name, info in data.items()
    if info.get("isCompiled") is True
}
print(f"Total files: {len(data)}")
print(f"Compiled files: {len(compiled)}")

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

# Sort by total_parameters (primary) and line_count (secondary), both descending
ranked = sorted(
    compiled.items(),
    key=lambda x: (x[1]["stats"]["total_parameters"], x[1]["line_count"]),
    reverse=True
)

selected = ranked[:TOP_N]

result = {name: info for name, info in selected}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=4)

print(f"\nSelected {len(result)} files")
print(f"Parameter range: {selected[-1][1]['stats']['total_parameters']} - {selected[0][1]['stats']['total_parameters']}")
print(f"Line count range: {min(v['line_count'] for v in result.values())} - {max(v['line_count'] for v in result.values())}")
print(f"Saved to: {OUTPUT_PATH}")
