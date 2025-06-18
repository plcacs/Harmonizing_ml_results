import json

json_files = ['gpt4O_stats_equal.json', 'deepseek_stats_equal.json', 'o1_mini_stats_equal.json']
error_count_files = ['llm_error_only_results_gpt4.json', 'llm_error_only_results_deepseek.json', 'llm_error_only_results_o1_mini.json']

def analyze_stats(stats_data, model_name):
    total_files = len(stats_data)
    files_with_errors = sum(1 for info in stats_data.values() if info.get('score', 0) > 0)
    files_without_errors = sum(1 for info in stats_data.values() if info.get('score', 0) == 0)

    time_with_errors = [info.get('time_taken', 0) for info in stats_data.values() if info.get('score', 0) > 0]
    avg_time_with_errors = sum(time_with_errors) / len(time_with_errors) if time_with_errors else 0

    time_without_errors = [info.get('time_taken', 0) for info in stats_data.values() if info.get('score', 0) == 0]
    avg_time_without_errors = sum(time_without_errors) / len(time_without_errors) if time_without_errors else 0

    annotation_diff_without_errors = [info.get('original_parameters_with_annotations', 0) - info.get('updated_parameters_with_annotations', 0) for info in stats_data.values() if info.get('score', 0) == 0]
    avg_annotation_diff_without_errors = sum(annotation_diff_without_errors) / len(annotation_diff_without_errors) if annotation_diff_without_errors else 0

    files_with_score_and_time = sum(1 for info in stats_data.values() if info.get('score', 0) > 0 and info.get('time_taken', 0) > 2000)

    print(f"\nAnalyzing {model_name}:")
    print(f"Total number of files: {total_files}")
    print(f"Number of files with errors (error_count > 0): {files_with_errors}")
    print(f"Average time taken for files with errors: {avg_time_with_errors:.2f}")
    print(f"Number of files without errors (error_count = 0): {files_without_errors}")
    print(f"Percentage of files without errors: {(files_without_errors/total_files)*100:.2f}%")
    print(f"Average time taken for files without errors: {avg_time_without_errors:.2f}")
    print(f"Average annotation difference for files without errors: {avg_annotation_diff_without_errors:.2f}")
    print(f"Number of files with score > 0 and time_taken > 2000: {files_with_score_and_time}")

for json_file in error_count_files:
    with open(json_file, 'r') as f:
        stats_data = json.load(f)
    analyze_stats(stats_data, json_file)

# Suggested statistics to analyze:
# 1. Total number of files
# 2. Number of files with error_count > 0
# 3. Average error count per file
# 4. Distribution of error counts (e.g., how many files have 1 error, 2 errors, etc.)
# 5. Number of files with each error keyword

# Distribution of error counts
