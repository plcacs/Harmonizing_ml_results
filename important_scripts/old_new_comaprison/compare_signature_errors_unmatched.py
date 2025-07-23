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


def find_unmatched_files(old_map, new_map):
    old_sigs = set(old_map.keys())
    new_sigs = set(new_map.keys())
    unmatched_old = old_sigs - new_sigs
    unmatched_new = new_sigs - old_sigs
    unmatched = {
        "unmatched_old": {str(sorted(list(sig))): old_map[sig] for sig in unmatched_old},
        "unmatched_new": {str(sorted(list(sig))): new_map[sig] for sig in unmatched_new},
    }
    return unmatched


def process_model(model_name, old_json, new_json, output_file):
    print(f"\nProcessing {model_name}...")
    old_data, new_data = load_data(old_json, new_json)
    old_sig_map = map_signatures(old_data)
    new_sig_map = map_signatures(new_data)
    unmatched = find_unmatched_files(old_sig_map, new_sig_map)
    with open(output_file, "w") as f:
        json.dump(unmatched, f, indent=4)
    print(f"Unmatched old signature groups: {len(unmatched['unmatched_old'])}")
    print(f"Unmatched new signature groups: {len(unmatched['unmatched_new'])}")
    print(f"Detailed unmatched results saved to {output_file}")


if __name__ == "__main__":
    models = [
        {
            "name": "o1-mini",
            "old_json": "o1-mini_code_similarity_old.json",
            "new_json": "o1-mini_code_similarity_new.json",
            "output": "signature_unmatched_results_o1-mini.json",
        },
        {
            "name": "gpt4o",
            "old_json": "gpt40_code_similarity_old.json",
            "new_json": "gpt40_code_similarity_new.json",
            "output": "signature_unmatched_results_gpt4o.json",
        },
        {
            "name": "deepseek",
            "old_json": "deepseek_code_similarity_old.json",
            "new_json": "deepseek_code_similarity_new.json",
            "output": "signature_unmatched_results_deepseek.json",
        },
    ]
    for model in models:
        process_model(
            model["name"], model["old_json"], model["new_json"], model["output"]
        ) 