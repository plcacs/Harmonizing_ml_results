import json
from collections import defaultdict


def load_data(old_path, new_path):
    with open(old_path) as f_old, open(new_path) as f_new:
        return json.load(f_old), json.load(f_new)


def map_signatures(data):
    sig_map = defaultdict(list)
    for filename, sig_list in data.items():
        sig = frozenset(sig_list)
        sig_map[sig].append(filename)
    return sig_map


def match_signatures(old_map, new_map):
    common = set(old_map.keys()) & set(new_map.keys())
    matched = []
    for sig in common:
        matched.append(
            {
                "file_signature": sorted(sig),
                "old_files": old_map[sig],
                "new_files": new_map[sig],
            }
        )
    return matched


def process_model(model_name, old_json, new_json, output_file):
    print(f"\nProcessing {model_name}...")
    old_data, new_data = load_data(old_json, new_json)
    old_sig_map = map_signatures(old_data)
    new_sig_map = map_signatures(new_data)
    matched = match_signatures(old_sig_map, new_sig_map)
    with open(output_file, "w") as f:
        json.dump(matched, f, indent=4)
    print(f"Total matched file_signature groups: {len(matched)}")
    print(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    models = [
        {
            "name": "o1-mini",
            "old_json": "o1-mini_code_similarity_old.json",
            "new_json": "o1-mini_code_similarity_new.json",
            "output": "signature_comparison_results_o1-mini.json",
        },
        {
            "name": "gpt4o",
            "old_json": "gpt40_code_similarity_old.json",
            "new_json": "gpt40_code_similarity_new.json",
            "output": "signature_comparison_results_gpt4o.json",
        },
        {
            "name": "deepseek",
            "old_json": "deepseek_code_similarity_old.json",
            "new_json": "deepseek_code_similarity_new.json",
            "output": "signature_comparison_results_deepseek.json",
        },
    ]
    for model in models:
        process_model(
            model["name"], model["old_json"], model["new_json"], model["output"]
        )
