import json

with open('deepseek_stats_equal.json', 'r') as f:
    deepseek_stats = json.load(f)

# Suggested statistics to analyze:
# 1. Total number of files
# 2. Number of files with error_count > 0
# 3. Average error count per file
# 4. Distribution of error counts (e.g., how many files have 1 error, 2 errors, etc.)
# 5. Number of files with each error keyword

print("\nAnalyzing deepseek_stats_equal.json:")
total_files = len(deepseek_stats)
files_with_errors = sum(1 for info in deepseek_stats.values() if info.get('error_count', 0) > 0)
avg_errors = sum(info.get('error_count', 0) for info in deepseek_stats.values()) / total_files if total_files > 0 else 0

print(f"Total files: {total_files}")
print(f"Files with errors: {files_with_errors}")
print(f"Average errors per file: {avg_errors:.2f}")

# Distribution of error counts
error_distribution = {}
for info in deepseek_stats.values():
    error_count = info.get('error_count', 0)
    error_distribution[error_count] = error_distribution.get(error_count, 0) + 1

print("\nError count distribution:")
for error_count, count in sorted(error_distribution.items()):
    print(f"{error_count} errors: {count} files")

# Count files with each error keyword
error_keywords = [
    'arg-type', 'incompatible', 'invalid type', 'missing return', 'optional',
    'union', 'return-value', 'unsupported operand'
]

error_keyword_counts = {k: 0 for k in error_keywords}
for info in deepseek_stats.values():
    errors = info.get('errors', [])
    for error in errors:
        error_lower = error.lower()
        for keyword in error_keywords:
            if keyword in error_lower:
                error_keyword_counts[keyword] += 1

print("\nFiles with each error keyword:")
for keyword, count in error_keyword_counts.items():
    print(f"{keyword}: {count} files") 