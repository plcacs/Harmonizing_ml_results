import json
import os

# Input files
MATCHED_PATHs = [
    "important_scripts/old_new_comaprison/matched_unmatched_gpt4o.json",
    "important_scripts/old_new_comaprison/matched_unmatched_o1-mini.json",
    "important_scripts/old_new_comaprison/matched_unmatched_deepseek.json",
]
SIGNATURES_PATHs = [
    "important_scripts/old_new_comaprison/gpt40_code_similarity_old.json",
    "important_scripts/old_new_comaprison/o1-mini_code_similarity_old.json",
    "important_scripts/old_new_comaprison/deepseek_code_similarity_old.json",
]
OUTPUT_PATHs = [
    "important_scripts/old_new_comaprison/Table_1_gen/signature_comparison_results_gpt4o.json",
    "important_scripts/old_new_comaprison/Table_1_gen/signature_comparison_results_o1-mini.json",
    "important_scripts/old_new_comaprison/Table_1_gen/signature_comparison_results_deepseek.json",
]

for MATCHED_PATH, SIGNATURES_PATH, OUTPUT_PATH in zip(
    MATCHED_PATHs, SIGNATURES_PATHs, OUTPUT_PATHs
):
    with open(MATCHED_PATH, "r") as f:
        matched_data = json.load(f)
    with open(SIGNATURES_PATH, "r") as f:
        signatures_data = json.load(f)

    total_matches = len(matched_data["matches"])
    total_signatures = len(signatures_data)
    matches_with_signature = sum(
        1 for old_file in matched_data["matches"] if old_file in signatures_data
    )
    # print(f"{MATCHED_PATH}: total matches = {total_matches}")
    # print(f"{SIGNATURES_PATH}: total signatures = {total_signatures}")
    # print(f"Matches with signature = {matches_with_signature}")

    # Load existing output if present
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    else:
        existing = []

    # Build a set of (old_file, new_file) pairs already present
    existing_pairs = set()
    for entry in existing:
        for old_file in entry.get("old_files", []):
            for new_file in entry.get("new_files", []):
                existing_pairs.add((old_file, new_file))
    count = 0
    results = existing[:]
    for old_file, new_file in matched_data["matches"].items():
        if (old_file, new_file) in existing_pairs:
            count += 1
            continue  # Skip redundant pairs
        file_signature = signatures_data.get(old_file)
        if file_signature is None:
            continue
        results.append(
            {
                "file_signature": file_signature,
                "old_files": [old_file],
                "new_files": [new_file],
            }
        )
        existing_pairs.add((old_file, new_file))  # Add to set to prevent future redundancy
    print(f"{OUTPUT_PATH}: {len(results)} entries")
    #print(f"Count of matches that are already in the output = {count}")
    with open(OUTPUT_PATH, "w") as f:
         json.dump(results, f, indent=2)
