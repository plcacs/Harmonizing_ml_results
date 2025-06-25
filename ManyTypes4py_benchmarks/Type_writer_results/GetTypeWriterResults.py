import json
import os


def process_model(mypy_file, stats_file, output_file):
    # Read the mypy results file
    with open(mypy_file, "r") as f:
        mypy_results = json.load(f)

    # Read the stats file
    with open(stats_file, "r") as f:
        stats_equal = json.load(f)

    # Find files where base_error_count == 0 but llm_error_count > 0
    target_files = {filename: data for filename, data in mypy_results.items()}
    print(len(target_files))
    # Create output dictionary with matching files from stats_equal
    output = {}
    for filename in target_files:
        if filename in stats_equal:
            output[filename] = stats_equal[filename]
    print(len(output))
    # Save the output to a new JSON file
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def main():
    # Process GPT4
    process_model(
        "merged_gpt4o.json",
        "gpt4O_stats_equal.json",
        "llm_error_only_results_gpt4.json",
    )

    # Process O1-mini
    process_model(
        "merged_o1-mini.json",
        "o1_mini_stats_equal.json",
        "llm_error_only_results_o1_mini.json",
    )

    # Process DeepSeek
    process_model(
        "merged_deepseek.json",
        "deepseek_stats_equal.json",
        "llm_error_only_results_deepseek.json",
    )


if __name__ == "__main__":
    main()
