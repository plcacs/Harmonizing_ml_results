import json
import os

def replace_path_suffix(input_file, output_file, old_suffix, new_prefix):
    """
    Replaces a specific path suffix with a new prefix in a JSON file and saves the result.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
        old_suffix: The suffix to replace (e.g., "D:/Projects/Datasets/many-types-4-py-dataset/data/cloned_repos/").
        new_prefix: The prefix to replace it with (e.g., "/home/C00454290/wahids_world/cloned_repos/").
    """
    try:
        with open(input_file, 'r') as f:
            json_data = json.load(f)

        def replace_in_path(path):
            converted_path = path.replace("\\", "/")
            if isinstance(converted_path, str) and converted_path.startswith(old_suffix):
                return new_prefix + "/" + converted_path[len(old_suffix):].lstrip("/") # Add missing slash and remove leading slash
            return converted_path

        def process_value(value):
            if isinstance(value, str):
                return replace_in_path(value)
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            elif isinstance(value, dict):
                return process_value(json.loads(json.dumps(value))) #recursive call
            else:
                return value

        new_data = {}
        for key, value in json_data.items():
            new_data[key] = process_value(value)

        with open(output_file, 'w') as f:
            json.dump(new_data, f, indent=4)

        print(f"Successfully replaced paths and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}' - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example Usage
input_filename = "filtered_python_files.json" # Replace with your input file name
output_filename = "filtered_python_files_linux.json" # Replace with your output file name
old_suffix = "D:/Projects/Datasets/many-types-4-py-dataset/data/cloned_repos/"
new_prefix = "/home/C00454290/wahids_world/cloned_repos"

# Create a sample input.json file for testing (if it doesn't exist

replace_path_suffix(input_filename, output_filename, old_suffix, new_prefix)
