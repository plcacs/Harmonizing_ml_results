import json
from collections import defaultdict

def load_data(old_path, new_path):
    with open(old_path) as f_old, open(new_path) as f_new:
        return json.load(f_old), json.load(f_new)

def map_signatures(data):
    sig_map = defaultdict(list)
    for filename, meta in data.items():
        sig = frozenset(meta["file_signature"])
        sig_map[sig].append(filename)
    return sig_map

def match_signatures(old_map, new_map, old_data, new_data):
    common = set(old_map.keys()) & set(new_map.keys())
    matched = []
    for sig in common:
        old_files_with_stats = []
        new_files_with_stats = []
        
        for old_file in old_map[sig]:
            old_files_with_stats.append({
                "filename": old_file,
                "stats": old_data[old_file]
            })
            
        for new_file in new_map[sig]:
            new_files_with_stats.append({
                "filename": new_file,
                "stats": new_data[new_file]
            })
            
        matched.append({
            "file_signature": sorted(sig),
            "old_files": old_files_with_stats,
            "new_files": new_files_with_stats
        })
    return matched

def compute_stats(matched, old_data, new_data):
    stats = {
        "both_error_count_0": 0,
        "old_0_new_>0": 0,
        "new_0_old_>0": 0,
        "both_error_count_>0": 0
    }

    for entry in matched:
        for old_file in entry["old_files"]:
            for new_file in entry["new_files"]:
                old_err = old_file["stats"]["error_count"]
                new_err = new_file["stats"]["error_count"]

                if old_err == 0 and new_err == 0:
                    stats["both_error_count_0"] += 1
                elif old_err == 0 and new_err > 0:
                    stats["old_0_new_>0"] += 1
                elif old_err > 0 and new_err == 0:
                    stats["new_0_old_>0"] += 1
                elif old_err > 0 and new_err > 0:
                    stats["both_error_count_>0"] += 1

    return stats

if __name__ == "__main__":
    old_json = "code_similarity_o1_mini_old.json"
    new_json = "code_similarity_o1_mini_new.json"

    old_data, new_data = load_data(old_json, new_json)
    old_sig_map = map_signatures(old_data)
    new_sig_map = map_signatures(new_data)
    matched = match_signatures(old_sig_map, new_sig_map, old_data, new_data)
    stats = compute_stats(matched, old_data, new_data)

    # Prepare detailed results
    results = {
        "stats": stats,
        "matched_files": matched
    }

    # Save results to JSON
    output_file = "signature_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Total matched file_signature groups: {len(matched)}")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"\nDetailed results saved to {output_file}")
