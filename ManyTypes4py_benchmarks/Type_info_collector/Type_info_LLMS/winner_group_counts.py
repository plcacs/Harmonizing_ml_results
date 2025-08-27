import csv
from collections import defaultdict
from typing import Dict, List, Tuple

# Inputs (relative to this script)
PER_FILE_CSV = "./precision_points_per_file.csv"

# Outputs
WINNER_GROUP_COUNTS_CSV = "./winner_group_counts.csv"
WINNER_GROUP_COUNTS_NORMALIZED_CSV = "./winner_group_counts_normalized.csv"


def read_per_file_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    rows = read_per_file_rows(PER_FILE_CSV)

    # counts[(k_winners)][llm] = count
    counts_by_group: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Track all LLM names we see in winners
    all_llms: set[str] = set()

    for row in rows:
        winners_field = row.get("winners", "").strip()
        if not winners_field:
            continue
        winners = [w.strip() for w in winners_field.split(";") if w.strip()]
        if not winners:
            continue
        k = len(winners)
        all_llms.update(winners)
        for llm in winners:
            counts_by_group[k][llm] += 1

    # Prepare flat rows for counts CSV
    counts_rows: List[Tuple[int, str, int]] = []
    for k in sorted(counts_by_group.keys()):
        for llm in sorted(all_llms):
            counts_rows.append((k, llm, counts_by_group[k].get(llm, 0)))

    with open(WINNER_GROUP_COUNTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["winner_group_size", "llm", "count"])
        writer.writerows(counts_rows)

    # Normalized per group (percentage per group for each LLM)
    normalized_rows: List[Tuple[int, str, str]] = []
    for k in sorted(counts_by_group.keys()):
        total_in_group = sum(counts_by_group[k].values())
        for llm in sorted(all_llms):
            c = counts_by_group[k].get(llm, 0)
            pct = (c * 100.0 / total_in_group) if total_in_group > 0 else 0.0
            normalized_rows.append((k, llm, f"{pct:.2f}%"))

    with open(WINNER_GROUP_COUNTS_NORMALIZED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["winner_group_size", "llm", "percent_of_group"])
        writer.writerows(normalized_rows)

    print(f"Wrote {WINNER_GROUP_COUNTS_CSV} and {WINNER_GROUP_COUNTS_NORMALIZED_CSV}")


if __name__ == "__main__":
    main()
