import csv
import os
import re
from collections import Counter, defaultdict
from typing import Tuple


TYPE_PATTERNS = [
    r"\bIncompatible\b",
    r"\bOptional\b",
    r"\breturn value type\b",
    r"\bargument[s]?\b.*\btype\b",
    r"\bassignment\b.*\btype\b",
    r"\bType\b",
    r"\bCallable\b",
    r"\bProtocol\b",
    r"\bTypedDict\b",
    r"\bvariance\b",
    r"\bTypeVar\b",
]

NONTYPE_PATTERNS = [
    r"\bName \".*\" is not defined\b",
    r"\bhas no attribute\b",
    r"\bCannot find (module|implementation|library stub)\b",
    r"\bModule \".*\" has no attribute\b",
    r"\bUnsupported operand\b",
]


def classify(message: str) -> str:
    msg = message or ""
    for pat in NONTYPE_PATTERNS:
        if re.search(pat, msg, re.IGNORECASE):
            return "non_type"
    for pat in TYPE_PATTERNS:
        if re.search(pat, msg, re.IGNORECASE):
            return "type"
    # Fallback: heuristic
    if "type" in msg.lower():
        return "type"
    return "non_type"


def main() -> None:
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "flip_results.csv")
    if not os.path.exists(csv_path):
        print("flip_results.csv not found; run flip_results.py first.")
        return

    counts = Counter()
    per_model = defaultdict(Counter)
    # Count on the failing run's top error (overall and per model)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "")
            run1_status = row.get("run1_status", "")
            run2_status = row.get("run2_status", "")
            if run1_status == run2_status:
                continue
            # Determine failing side and use its top error
            if run1_status == "fail" and run2_status == "success":
                msg = row.get("run1_error_top", "")
            elif run1_status == "success" and run2_status == "fail":
                msg = row.get("run2_error_top", "")
            else:
                # unexpected
                msg = row.get("run2_error_top", "") or row.get("run1_error_top", "")
            bucket = classify(msg)
            counts[bucket] += 1
            per_model[model][bucket] += 1

    # Overall
    print(f"type_related={counts.get('type', 0)} non_type_related={counts.get('non_type', 0)} total={sum(counts.values())}")
    # Per model concise lines
    for model in sorted(per_model.keys()):
        m = per_model[model]
        print(f"{model}: type_related={m.get('type', 0)} non_type_related={m.get('non_type', 0)} total={sum(m.values())}")


if __name__ == "__main__":
    main()


