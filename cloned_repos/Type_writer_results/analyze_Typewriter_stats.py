import json

with open('gpt4O_stats_equal.json', 'r') as f:
    deepseek_stats = json.load(f)

# Suggested statistics to analyze:
# 1. Total number of files
# 2. Number of files with error_count > 0
# 3. Average error count per file
# 4. Distribution of error counts (e.g., how many files have 1 error, 2 errors, etc.)
# 5. Number of files with each error keyword

# Calculate statistics
total_files = len(deepseek_stats)
files_with_errors = sum(1 for info in deepseek_stats.values() if info.get('score', 0) > 0)
files_without_errors = sum(1 for info in deepseek_stats.values() if info.get('score', 0) == 0)

# Calculate average time for files with errors
time_with_errors = [info.get('time_taken', 0) for info in deepseek_stats.values() if info.get('score', 0) > 0]
avg_time_with_errors = sum(time_with_errors) / len(time_with_errors) if time_with_errors else 0

# Calculate average time for files without errors
time_without_errors = [info.get('time_taken', 0) for info in deepseek_stats.values() if info.get('score', 0) == 0]
avg_time_without_errors = sum(time_without_errors) / len(time_without_errors) if time_without_errors else 0

# Calculate average annotation difference for files without errors
annotation_diff_without_errors = [info.get('original_parameters_with_annotations', 0) - info.get('updated_parameters_with_annotations', 0) for info in deepseek_stats.values() if info.get('score', 0) == 0]
avg_annotation_diff_without_errors = sum(annotation_diff_without_errors) / len(annotation_diff_without_errors) if annotation_diff_without_errors else 0

# Calculate number of files where score > 0 and time_taken > 2000
files_with_score_and_time = sum(1 for info in deepseek_stats.values() if info.get('score', 0) > 0 and info.get('time_taken', 0) > 2000)

# Print statistics
print("\nAnalyzing deepseek_stats_equal.json:")
print(f"Total number of files: {total_files}")
print(f"Number of files with errors (error_count > 0): {files_with_errors}")
print(f"Average time taken for files with errors: {avg_time_with_errors:.2f}")
print(f"Number of files without errors (error_count = 0): {files_without_errors}")
print(f"Average time taken for files without errors: {avg_time_without_errors:.2f}")
print(f"Average annotation difference for files without errors: {avg_annotation_diff_without_errors:.2f}")
print(f"Number of files with score > 0 and time_taken > 2000: {files_with_score_and_time}")

# Distribution of error counts
