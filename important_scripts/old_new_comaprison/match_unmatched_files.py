import json
import re
from collections import defaultdict

def extract_base(filename):
    # Remove .py extension anywhere
    name = filename.replace('.py', '')
    # Remove model tags and trailing hashes (e.g., _gpt4_xxx, _7dee21, etc.)
    name = re.sub(r'(_deepseek_\w+|_[0-9a-f]{6,}|_[a-z]{4,}\d*)$', '', name)
    # Remove any remaining trailing underscores
    name = name.rstrip('_')
    return name

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0

def load_unmatched(path):
    with open(path) as f:
        data = json.load(f)
    return data['unmatched_old'], data['unmatched_new']

def parse_sig_map(sig_map):
    # sig_map: {signature_str: [file1, file2, ...]}
    file_to_sig = {}
    for sig_str, files in sig_map.items():
        sig_set = set(eval(sig_str))
        for f in files:
            file_to_sig[f] = sig_set
    return file_to_sig

def main():
    input_path = 'signature_unmatched_results_deepseek.json'
    output_path = 'matched_unmatched_deepseek.json'
    unmatched_old, unmatched_new = load_unmatched(input_path)
    old_files = parse_sig_map(unmatched_old)
    new_files = parse_sig_map(unmatched_new)

    # Build base name to files mapping for new
    base_to_new = defaultdict(list)
    for f in new_files:
        base = extract_base(f)
        base_to_new[base].append(f)

    matches = {}
    unmatched = []
    for old_f, old_sig in old_files.items():
        base = extract_base(old_f)
        candidates = base_to_new.get(base, [])
        if not candidates:
            unmatched.append(old_f)
            continue
        if len(candidates) == 1:
            matches[old_f] = candidates[0]
        else:
            # Use signature similarity
            best = None
            best_score = -1
            for cand in candidates:
                score = jaccard_similarity(old_sig, new_files[cand])
                if score > best_score:
                    best_score = score
                    best = cand
            matches[old_f] = best
    with open(output_path, 'w') as f:
        json.dump({'matches': matches, 'unmatched_old': unmatched}, f, indent=2)
    print(f"Done. Matches: {len(matches)}, Unmatched: {len(unmatched)}. Output: {output_path}")

if __name__ == '__main__':
    main() 