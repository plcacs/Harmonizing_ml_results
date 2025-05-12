import json

print("Deepseek")
with open('mypy_results_gpt4o_with_errors.json', 'r') as f:
    data_gpt4o = json.load(f)

with open('mypy_results_no_type.json', 'r') as f:
    data_no_type = json.load(f)

# Find files with error_count==0 in no_type but not in gpt4O
eligible_files = set()
for filename, info in data_no_type.items():
    if info.get('error_count', 0) == 0 and filename in data_gpt4o:
        if data_gpt4o[filename].get('error_count', 0) > 0:
            eligible_files.add(filename)

error_keywords = [
    'arg-type', 'incompatible', 'invalid type', 'missing return', 'optional',
    'union', 'return-value', 'unsupported operand'
]

# Collect matching files and the matched keyword
matching = []

# New: Track counts per keyword
files_per_keyword = {k: set() for k in error_keywords}
errors_per_keyword = {k: 0 for k in error_keywords}

for filename in eligible_files:
    info = data_gpt4o[filename]
    errors = info.get('errors', [])
    for error in errors:
        error_lower = error.lower()
        for keyword in error_keywords:
            if keyword in error_lower:
                matching.append((filename, keyword))
                files_per_keyword[keyword].add(filename)
                errors_per_keyword[keyword] += 1
                # Only first matching keyword per file
        else:
            continue
    

# Show just a few for manual inspection
print("Sample files with matching error keywords:")
for file, keyword in matching[:10]:
    print(f"{file}: matched keyword -> {keyword}")

# Print summary statistics
print("\nSummary per error keyword:")
for keyword in error_keywords:
    print(f"{keyword}: {len(files_per_keyword[keyword])} files, {errors_per_keyword[keyword]} total errors")
