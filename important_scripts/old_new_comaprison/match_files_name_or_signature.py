import json
from collections import defaultdict
import os


def load_data(old_path, new_path):
    with open(old_path) as f_old, open(new_path) as f_new:
        return json.load(f_old), json.load(f_new)


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0


def map_signatures(data):
    sig_map = {}
    for filename, sig_list in data.items():
        sig_map[filename] = set(sig_list)
    return sig_map


def match_files_by_fuzzy_signature(old_data, new_data, threshold=0.5):
    old_sig_map = map_signatures(old_data)
    new_sig_map = map_signatures(new_data)
    matched = []
    unmatched_old = []
    matched_new_files = set()
    for old_file, old_sig in old_sig_map.items():
        best_score = 0
        best_matches = []
        for new_file, new_sig in new_sig_map.items():
            score = jaccard_similarity(old_sig, new_sig)
            if score > best_score:
                best_score = score
                best_matches = [(new_file, score)]
            elif score == best_score and score > 0:
                best_matches.append((new_file, score))
        if best_score >= threshold:
            for new_file, score in best_matches:
                matched.append(
                    {
                        "old": old_file,
                        "new": new_file,
                        "match_type": "fuzzy_signature",
                        "similarity": score,
                    }
                )
                matched_new_files.add(new_file)
        else:
            unmatched_old.append(old_file)
    unmatched_new = [f for f in new_sig_map if f not in matched_new_files]
    return matched, unmatched_old, unmatched_new


def process_model(model_name, old_json, new_json, output_file, threshold=0.5):
    print(f"\nProcessing {model_name}...")
    old_data, new_data = load_data(old_json, new_json)
    matched, unmatched_old, unmatched_new = match_files_by_fuzzy_signature(
        old_data, new_data, threshold
    )
    result = {
        "matched": matched,
        "unmatched_old": unmatched_old,
        "unmatched_new": unmatched_new,
    }
    # with open(output_file, "w") as f:
    #    json.dump(result, f, indent=4)
    print(f"Matched: {len(matched)} (threshold={threshold})")
    print(f"Unmatched old: {len(unmatched_old)}")
    print(f"Unmatched new: {len(unmatched_new)}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    threshold = 0.5  # You can adjust this value for stricter/looser matching
    models = [
        {
            "name": "o1-mini",
            "old_json": "o1-mini_code_similarity_old.json",
            "new_json": "o1-mini_code_similarity_new.json",
            "output": "matched_group/matched_unmatched_o1-mini.json",
        },
        {
            "name": "gpt4o",
            "old_json": "gpt40_code_similarity_old.json",
            "new_json": "gpt40_code_similarity_new.json",
            "output": "matched_group/matched_unmatched_gpt4o.json",
        },
        {
            "name": "deepseek",
            "old_json": "deepseek_code_similarity_old.json",
            "new_json": "deepseek_code_similarity_new.json",
            "output": "matched_group/matched_unmatched_deepseek.json",
        },
    ]
    for model in models:
        process_model(
            model["name"],
            model["old_json"],
            model["new_json"],
            model["output"],
            threshold,
        )
