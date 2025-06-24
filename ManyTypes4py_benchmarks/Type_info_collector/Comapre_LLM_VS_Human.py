import json
from typing import Dict, List, Tuple, Any


def load_annotations(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(file_path, "r") as f:
        data = json.load(f)

    # The data structure is: filename -> {function_signature -> [annotations]}
    # We need to flatten this to: filename -> [all_annotations]
    flattened_data = {}

    for filename, functions in data.items():
        all_annotations = []
        if isinstance(functions, dict):
            for func_sig, annotations in functions.items():
                if isinstance(annotations, list):
                    all_annotations.extend(annotations)
                elif isinstance(annotations, dict):
                    all_annotations.append(annotations)
                elif isinstance(annotations, str):
                    all_annotations.append(
                        {"category": "unknown", "type": [annotations]}
                    )
        elif isinstance(functions, list):
            all_annotations = functions
        elif isinstance(functions, str):
            all_annotations = [{"category": "unknown", "type": [functions]}]

        flattened_data[filename] = all_annotations

    return flattened_data


def normalize_type(type_str: str) -> str:
    if not type_str:
        return "dyn"
    return type_str.replace(" ", "").lower()


def type_level_exact_match(ann1, ann2) -> Tuple[int, int]:
    exact = total = 0

    # Find common filenames
    common_files = set(ann1.keys()) & set(ann2.keys())

    for filename in common_files:
        items1 = ann1[filename]
        items2 = ann2[filename]

        if not isinstance(items1, list) or not isinstance(items2, list):
            continue

        # Match annotations by category and name
        for item1 in items1:
            if not isinstance(item1, dict):
                continue

            for item2 in items2:
                if not isinstance(item2, dict):
                    continue

                # Match by category and name
                if item1.get("category") == item2.get("category") and item1.get(
                    "name"
                ) == item2.get("name"):

                    total += 1

                    # Get types safely
                    t1 = item1.get("type", [])
                    t2 = item2.get("type", [])

                    if (
                        isinstance(t1, list)
                        and isinstance(t2, list)
                        and len(t1) > 0
                        and len(t2) > 0
                        and isinstance(t1[0], str)
                        and isinstance(t2[0], str)
                    ):

                        if normalize_type(t1[0]) == normalize_type(t2[0]):
                            exact += 1
                    break  # Found match, move to next item1

    return exact, total


def top_level_type(typ: str) -> str:
    if not typ:
        return "dyn"
    return typ.split("[")[0] if "[" in typ else typ


def top_level_type_match(ann1, ann2) -> Tuple[int, int]:
    matched = total = 0

    # Find common filenames
    common_files = set(ann1.keys()) & set(ann2.keys())

    for filename in common_files:
        items1 = ann1[filename]
        items2 = ann2[filename]

        if not isinstance(items1, list) or not isinstance(items2, list):
            continue

        # Match annotations by category and name
        for item1 in items1:
            if not isinstance(item1, dict):
                continue

            for item2 in items2:
                if not isinstance(item2, dict):
                    continue

                # Match by category and name
                if item1.get("category") == item2.get("category") and item1.get(
                    "name"
                ) == item2.get("name"):

                    total += 1

                    # Get types safely
                    t1 = item1.get("type", [])
                    t2 = item2.get("type", [])

                    if (
                        isinstance(t1, list)
                        and isinstance(t2, list)
                        and len(t1) > 0
                        and len(t2) > 0
                        and isinstance(t1[0], str)
                        and isinstance(t2[0], str)
                    ):

                        if top_level_type(t1[0]) == top_level_type(t2[0]):
                            matched += 1
                    break  # Found match, move to next item1

    return matched, total


def coverage_difference(ann1, ann2) -> Tuple[int, int]:
    len1 = sum(len(v) for v in ann1.values() if isinstance(v, list))
    len2 = sum(len(v) for v in ann2.values() if isinstance(v, list))
    return len1, len2


def dynamic_static_pref(ann) -> Tuple[int, int]:
    dyn = stat = 0
    for entries in ann.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                types = entry.get("type", [])
                if types and isinstance(types, list) and len(types) > 0:
                    type_str = types[0]
                    if isinstance(type_str, str):
                        if normalize_type(type_str) == "dyn" or not type_str.strip():
                            dyn += 1
                        else:
                            stat += 1
    return stat, dyn


def compare_json_annotations(file1: str, file2: str):
    try:
        ann1 = load_annotations(file1)
        ann2 = load_annotations(file2)

        print(f"Loaded {len(ann1)} files from {file1} (Human annotated)")
        print(f"Loaded {len(ann2)} files from {file2} (LLM annotated)")

        exact, total = type_level_exact_match(ann1, ann2)
        print(
            f"Type-Level Exact Match (Human vs LLM): {exact}/{total} ({(exact/total*100 if total else 0):.2f}%)"
        )

        top_match, top_total = top_level_type_match(ann1, ann2)
        print(
            f"Top-Level Type Match (Human vs LLM): {top_match}/{top_total} ({(top_match/top_total*100 if top_total else 0):.2f}%)"
        )

        # len1, len2 = coverage_difference(ann1, ann2)
        # print(f"Annotation Coverage: File1={len1}, File2={len2}")

        stat1, dyn1 = dynamic_static_pref(ann1)
        stat2, dyn2 = dynamic_static_pref(ann2)
        print(
            f"Dynamic vs Static (Human): Static={stat1}, Dynamic={dyn1} ({(stat1/(stat1+dyn1)*100 if stat1+dyn1 else 0):.2f}%)"
        )
        print(
            f"Dynamic vs Static (LLM): Static={stat2}, Dynamic={dyn2} ({(stat2/(stat2+dyn2)*100 if stat2+dyn2 else 0):.2f}%)"
        )

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    print("GPT-4o")

    compare_json_annotations(
        "Type_info_original_files.json", "Type_info_gpt4o_benchmarks.json"
    )
    print("--------------------------------")
    print("O1-mini")
    compare_json_annotations(
        "Type_info_original_files.json", "Type_info_o1_mini_benchmarks.json"
    )
    print("--------------------------------")
    print("DeepSeek")
    compare_json_annotations(
        "Type_info_original_files.json", "Type_info_deep_seek_benchmarks.json"
    )
