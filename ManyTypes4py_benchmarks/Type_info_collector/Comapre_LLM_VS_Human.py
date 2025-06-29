import json
from typing import Dict, List, Tuple, Any


def load_annotations(file_path: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    with open(file_path, "r") as f:
        data = json.load(f)

    # Return the original structure: filename -> {function_signature -> [annotations]}
    return data


def normalize_type(type_str: str) -> str:
    if not type_str:
        return "dyn"
    # Remove quotes and spaces, then convert to lowercase
    return type_str.replace(" ", "").replace("'", "").replace('"', "").lower()


def type_level_exact_match(ann1, ann2) -> Tuple[int, int]:
    exact = total = 0
    file_count = 0
    comparison_details = []
    json_output = {}
    # Find common filenames
    common_files = set(ann1.keys()) & set(ann2.keys())

    for filename in common_files:
        items1 = ann1[filename]
        items2 = ann2[filename]
        is_file_has_type = False
        file_comparisons = []
        file_json_data = {}

        if not isinstance(items1, dict) or not isinstance(items2, dict):
            continue

        # Find common function signatures
        common_funcs = set(items1.keys()) & set(items2.keys())

        for func_sig in common_funcs:
            func_items1 = items1[func_sig]
            func_items2 = items2[func_sig]
            func_json_data = []

            if not isinstance(func_items1, list) or not isinstance(func_items2, list):
                continue

            # Match annotations by category and name within this function
            for item1 in func_items1:
                if not isinstance(item1, dict):
                    continue

                for item2 in func_items2:
                    if not isinstance(item2, dict):
                        continue

                    # Match by category and name
                    if item1.get("category") == item2.get("category") and item1.get(
                        "name"
                    ) == item2.get("name"):

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
                            # Skip empty, whitespace-only, "None", or "any/Any" types
                            if (
                                not t1[0].strip()
                                or not t2[0].strip()
                                or t1[0].strip().lower() == "none"
                                or t2[0].strip().lower() == "none"
                                or t1[0].strip().lower() == "any"
                                or t2[0].strip().lower() == "any"
                            ):
                                continue

                            total += 1
                            is_file_has_type = True

                            is_match = normalize_type(t1[0]) == normalize_type(t2[0])
                            if is_match:
                                exact += 1

                            # Store comparison details for console output
                            file_comparisons.append(
                                {
                                    "category": item1.get("category"),
                                    "name": item1.get("name"),
                                    "human_type": t1[0],
                                    "llm_type": t2[0],
                                    "match": is_match,
                                }
                            )

                            # Store JSON data
                            func_json_data.append(
                                {
                                    "category": item1.get("category", "unknown"),
                                    "param_name": item1.get("name", "unknown"),
                                    "Human": t1[0],
                                    "LLM": t2[0],
                                    "match": is_match,
                                }
                            )
                        break  # Found match, move to next item1

            if func_json_data:
                file_json_data[func_sig] = func_json_data

        if is_file_has_type:
            file_count += 1
            if len(comparison_details) < 10:
                comparison_details.append(
                    {"filename": filename, "comparisons": file_comparisons}
                )

            # Add to JSON output
            if file_json_data:
                json_output[filename] = file_json_data

    # print(f"Number of files with type annotations: {file_count}")

    # Print first 10 files and their parameters
    """print(f"\nFirst {len(comparison_details)} files used for comparison:")
    for file_detail in comparison_details:
        print(f"\nFile: {file_detail['filename']}")
        for comp in file_detail['comparisons']:
            match_status = "✓" if comp['match'] else "✗"
            print(f"  {match_status} {comp['category']} - {comp['name']}: Human='{comp['human_type']}' vs LLM='{comp['llm_type']}'")
    """
    return exact, total, json_output


def top_level_type(typ: str) -> str:
    if not typ:
        return "dyn"

    # Handle Optional[type] -> extract inner type
    if typ.startswith("Optional[") and typ.endswith("]"):
        return typ[9:-1]  # Remove 'Optional[' and ']'

    # Handle Union types like "str | None" -> extract first type
    if " | " in typ:
        return typ.split(" | ")[0]

    # Handle generic types like List[str] -> extract base type
    return typ.split("[")[0] if "[" in typ else typ


def top_level_type_match(ann1, ann2) -> Tuple[int, int]:
    matched = total = 0
    comparison_details = []
    json_output = {}
    # Find common filenames
    common_files = set(ann1.keys()) & set(ann2.keys())

    for filename in common_files:
        items1 = ann1[filename]
        items2 = ann2[filename]
        file_comparisons = []
        file_json_data = {}

        if not isinstance(items1, dict) or not isinstance(items2, dict):
            continue

        # Find common function signatures
        common_funcs = set(items1.keys()) & set(items2.keys())

        for func_sig in common_funcs:
            func_items1 = items1[func_sig]
            func_items2 = items2[func_sig]
            func_json_data = []

            if not isinstance(func_items1, list) or not isinstance(func_items2, list):
                continue

            # Match annotations by category and name within this function
            for item1 in func_items1:
                if not isinstance(item1, dict):
                    continue

                for item2 in func_items2:
                    if not isinstance(item2, dict):
                        continue

                    # Match by category and name
                    if item1.get("category") == item2.get("category") and item1.get(
                        "name"
                    ) == item2.get("name"):

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
                            # Skip empty, whitespace-only, or "None" types (same as exact match)
                            if (
                                not t1[0].strip()
                                or not t2[0].strip()
                                or t1[0].strip().lower() == "none"
                                or t2[0].strip().lower() == "none"
                            ):
                                continue

                            total += 1

                            # Get top-level types and normalize them
                            top1 = normalize_type(top_level_type(t1[0]))
                            top2 = normalize_type(top_level_type(t2[0]))

                            is_match = top1 == top2
                            if is_match:
                                matched += 1

                            # Store comparison details for console output
                            file_comparisons.append(
                                {
                                    "category": item1.get("category"),
                                    "name": item1.get("name"),
                                    "human_type": t1[0],
                                    "llm_type": t2[0],
                                    "human_top": top_level_type(t1[0]),
                                    "llm_top": top_level_type(t2[0]),
                                    "match": is_match,
                                }
                            )

                            # Store JSON data
                            func_json_data.append(
                                {
                                    "category": item1.get("category", "unknown"),
                                    "param_name": item1.get("name", "unknown"),
                                    "Human": t1[0],
                                    "LLM": t2[0],
                                    "Human_top": top_level_type(t1[0]),
                                    "LLM_top": top_level_type(t2[0]),
                                    "match": is_match,
                                }
                            )
                        break  # Found match, move to next item1

            if func_json_data:
                file_json_data[func_sig] = func_json_data

        if file_comparisons and len(comparison_details) < 10:
            comparison_details.append(
                {"filename": filename, "comparisons": file_comparisons}
            )

        # Add to JSON output
        if file_json_data:
            json_output[filename] = file_json_data

    # Print first 10 files and their parameters
    """print(f"\nFirst {len(comparison_details)} files used for top-level comparison:")
    for file_detail in comparison_details:
        print(f"\nFile: {file_detail['filename']}")
        for comp in file_detail['comparisons']:
            match_status = "✓" if comp['match'] else "✗"
            print(f"  {match_status} {comp['category']} - {comp['name']}: Human='{comp['human_type']}'({comp['human_top']}) vs LLM='{comp['llm_type']}'({comp['llm_top']})")
    """

    return matched, total, json_output


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


def count_any_types(ann) -> int:
    any_count = 0
    for filename, functions in ann.items():
        if isinstance(functions, dict):
            for func_sig, annotations in functions.items():
                if isinstance(annotations, list):
                    for entry in annotations:
                        if isinstance(entry, dict):
                            types = entry.get("type", [])
                            if types and isinstance(types, list) and len(types) > 0:
                                type_str = types[0]
                                if isinstance(type_str, str):
                                    # Check for Any in various forms
                                    normalized_type = normalize_type(type_str)
                                    if "any" in normalized_type:
                                        any_count += 1
    return any_count


def count_any_intersection(ann1, ann2) -> int:
    """
    Count how many parameters that humans typed as Any are also typed as Any by LLM
    Returns: intersection_count
    """
    intersection_count = 0

    # Find common filenames
    common_files = set(ann1.keys()) & set(ann2.keys())

    for filename in common_files:
        functions1 = ann1[filename]
        functions2 = ann2[filename]

        if not isinstance(functions1, dict) or not isinstance(functions2, dict):
            continue

        # Find common function signatures
        common_funcs = set(functions1.keys()) & set(functions2.keys())

        for func_sig in common_funcs:
            items1 = functions1[func_sig]
            items2 = functions2[func_sig]

            if not isinstance(items1, list) or not isinstance(items2, list):
                continue

            # Match annotations by category and name
            for item1 in items1:
                if not isinstance(item1, dict):
                    continue

                # Check if human typed this as Any
                types1 = item1.get("type", [])
                if types1 and isinstance(types1, list) and len(types1) > 0:
                    type_str1 = types1[0]
                    if isinstance(type_str1, str):
                        normalized_type1 = normalize_type(type_str1)
                        if "any" in normalized_type1:

                            # Find corresponding LLM annotation
                            for item2 in items2:
                                if not isinstance(item2, dict):
                                    continue

                                # Match by category and name
                                if item1.get("category") == item2.get(
                                    "category"
                                ) and item1.get("name") == item2.get("name"):

                                    # Check if LLM also typed this as Any
                                    types2 = item2.get("type", [])
                                    if (
                                        types2
                                        and isinstance(types2, list)
                                        and len(types2) > 0
                                    ):
                                        type_str2 = types2[0]
                                        if isinstance(type_str2, str):
                                            normalized_type2 = normalize_type(type_str2)
                                            if "any" in normalized_type2:
                                                intersection_count += 1
                                    break  # Found match, move to next item1

    return intersection_count


def is_simple_type(type_str: str) -> bool:
    """Check if type is simple (primitive) or complex"""
    simple_types = {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "complex",
        "none",
        "none_type",
    }
    normalized = normalize_type(type_str)

    # Check if it's a simple type
    if normalized in simple_types:
        return True

    # Check if it's Optional[simple_type]
    if normalized.startswith("optional[") and normalized.endswith("]"):
        inner_type = normalized[9:-1]  # Remove 'optional[' and ']'
        if inner_type in simple_types:
            return True

    return False


def get_type_complexity(type_str: str) -> int:
    """Calculate type complexity based on nesting depth and structure"""
    if not type_str:
        return 0

    complexity = 0
    normalized = normalize_type(type_str)

    # Count brackets (indicates nesting)
    complexity += normalized.count("[") + normalized.count("(")

    # Count commas (indicates multiple types)
    complexity += normalized.count(",")

    # Bonus complexity for complex patterns
    if "union[" in normalized or "optional[" in normalized:
        complexity += 1

    if "dict[" in normalized or "list[" in normalized or "set[" in normalized:
        complexity += 1

    return complexity


def analyze_type_categories_and_complexity(ann1, ann2) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Analyze type categories and complexity for the same parameters
    Returns: (human_categories, llm_categories, human_complexity, llm_complexity)
    """
    human_simple = human_complex = 0
    llm_simple = llm_complex = 0
    human_complexity_scores = []
    llm_complexity_scores = []

    # Find common filenames
    common_files = set(ann1.keys()) & set(ann2.keys())

    for filename in common_files:
        functions1 = ann1[filename]
        functions2 = ann2[filename]

        if not isinstance(functions1, dict) or not isinstance(functions2, dict):
            continue

        # Find common function signatures
        common_funcs = set(functions1.keys()) & set(functions2.keys())

        for func_sig in common_funcs:
            items1 = functions1[func_sig]
            items2 = functions2[func_sig]

            if not isinstance(items1, list) or not isinstance(items2, list):
                continue

            # Match annotations by category and name
            for item1 in items1:
                if not isinstance(item1, dict):
                    continue

                # Find corresponding LLM annotation
                for item2 in items2:
                    if not isinstance(item2, dict):
                        continue

                    # Match by category and name
                    if item1.get("category") == item2.get("category") and item1.get(
                        "name"
                    ) == item2.get("name"):

                        # Get types safely
                        types1 = item1.get("type", [])
                        types2 = item2.get("type", [])

                        if (
                            types1
                            and isinstance(types1, list)
                            and len(types1) > 0
                            and types2
                            and isinstance(types2, list)
                            and len(types2) > 0
                        ):

                            type_str1 = types1[0]
                            type_str2 = types2[0]

                            if isinstance(type_str1, str) and isinstance(
                                type_str2, str
                            ):
                                # Analyze human type
                                if is_simple_type(type_str1):
                                    human_simple += 1
                                else:
                                    human_complex += 1

                                # Analyze LLM type
                                if is_simple_type(type_str2):
                                    llm_simple += 1
                                else:
                                    llm_complex += 1

                                # Calculate complexity scores
                                human_complexity_scores.append(
                                    get_type_complexity(type_str1)
                                )
                                llm_complexity_scores.append(
                                    get_type_complexity(type_str2)
                                )

                        break  # Found match, move to next item1

    human_categories = {"simple": human_simple, "complex": human_complex}
    llm_categories = {"simple": llm_simple, "complex": llm_complex}

    human_complexity = {
        "avg": (
            sum(human_complexity_scores) / len(human_complexity_scores)
            if human_complexity_scores
            else 0
        ),
        "max": max(human_complexity_scores) if human_complexity_scores else 0,
        "min": min(human_complexity_scores) if human_complexity_scores else 0,
    }

    llm_complexity = {
        "avg": (
            sum(llm_complexity_scores) / len(llm_complexity_scores)
            if llm_complexity_scores
            else 0
        ),
        "max": max(llm_complexity_scores) if llm_complexity_scores else 0,
        "min": min(llm_complexity_scores) if llm_complexity_scores else 0,
    }

    return human_categories, llm_categories, human_complexity, llm_complexity


def compare_json_annotations(file1: str, file2: str, llm_name):
    try:
        ann1 = load_annotations(file1)
        ann2 = load_annotations(file2)

        # print(f"Loaded {len(ann1)} files from {file1} (Human annotated)")
        # print(f"Loaded {len(ann2)} files from {file2} (LLM annotated)")

        exact, total, json_output = type_level_exact_match(ann1, ann2)
        print(
            f"Type-Level Exact Match (Human vs LLM): {exact}/{total} ({(exact/total*100 if total else 0):.2f}%)"
        )

        # Save JSON output
        output_filename = f"type_comparison_{llm_name}.json"
        with open(output_filename, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON comparison saved to: {output_filename}")

        top_match, top_total, top_json_output = top_level_type_match(ann1, ann2)
        print(
            f"Top-Level Type Match (Human vs LLM): {top_match}/{top_total} ({(top_match/top_total*100 if top_total else 0):.2f}%)"
        )

        # Save top-level JSON output
        top_output_filename = f"top_level_comparison_{llm_name}.json"
        with open(top_output_filename, "w") as f:
            json.dump(top_json_output, f, indent=2)
        print(f"Top-level JSON comparison saved to: {top_output_filename}")

        # len1, len2 = coverage_difference(ann1, ann2)
        # print(f"Annotation Coverage: File1={len1}, File2={len2}")

        """stat1, dyn1 = dynamic_static_pref(ann1)
        stat2, dyn2 = dynamic_static_pref(ann2)
        print(
            f"Dynamic vs Static (Human): Static={stat1}, Dynamic={dyn1} ({(stat1/(stat1+dyn1)*100 if stat1+dyn1 else 0):.2f}%)"
        )
        print(
            f"Dynamic vs Static (LLM): Static={stat2}, Dynamic={dyn2} ({(stat2/(stat2+dyn2)*100 if stat2+dyn2 else 0):.2f}%)"
        )"""

        # Count Any types
        any_count1 = count_any_types(ann1)
        any_count2 = count_any_types(ann2)

        # Calculate total annotations for nested structure
        total_annotations1 = 0
        total_annotations2 = 0

        for filename, functions in ann1.items():
            if isinstance(functions, dict):
                for func_sig, annotations in functions.items():
                    if isinstance(annotations, list):
                        total_annotations1 += len(annotations)

        for filename, functions in ann2.items():
            if isinstance(functions, dict):
                for func_sig, annotations in functions.items():
                    if isinstance(annotations, list):
                        total_annotations2 += len(annotations)

        print(
            f"Any Type Usage (Human): {any_count1}/{total_annotations1} ({(any_count1/total_annotations1*100 if total_annotations1 else 0):.2f}%)"
        )
        print(
            f"Any Type Usage (LLM): {any_count2}/{total_annotations2} ({(any_count2/total_annotations2*100 if total_annotations2 else 0):.2f}%)"
        )

        # Count intersection of Any types
        intersection_count = count_any_intersection(ann1, ann2)
        print(
            f"Any Type Intersection: {intersection_count}/{any_count1} human Any types are also Any in LLM ({(intersection_count/any_count1*100 if any_count1 else 0):.2f}%)"
        )

        # Analyze type categories and complexity
        human_cats, llm_cats, human_comp, llm_comp = (
            analyze_type_categories_and_complexity(ann1, ann2)
        )
        total_matched = human_cats["simple"] + human_cats["complex"]

        print(f"\nType Category Distribution (for {total_matched} matched parameters):")
        print(
            f"Human: Simple={human_cats['simple']}, Complex={human_cats['complex']} ({(human_cats['simple']/total_matched*100 if total_matched else 0):.2f}% simple)"
        )
        print(
            f"LLM:   Simple={llm_cats['simple']}, Complex={llm_cats['complex']} ({(llm_cats['simple']/total_matched*100 if total_matched else 0):.2f}% simple)"
        )

        # print(f"\nType Complexity Analysis (for {total_matched} matched parameters):")
        # print(f"Human: Avg={human_comp['avg']:.2f}, Max={human_comp['max']}, Min={human_comp['min']}")
        # print(f"LLM:   Avg={llm_comp['avg']:.2f}, Max={llm_comp['max']}, Min={llm_comp['min']}")

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    print("GPT-4o")

    compare_json_annotations(
        "Type_info_original_files.json", "Type_info_gpt4o_benchmarks.json", "gpt4o"
    )
    print("--------------------------------")
    print("O1-mini")
    compare_json_annotations(
        "Type_info_original_files.json",
        "Type_info_o1_mini_benchmarks.json",
        "o1-mini",
    )
    print("--------------------------------")
    print("DeepSeek")
    compare_json_annotations(
        "Type_info_original_files.json",
        "Type_info_deep_seek_benchmarks.json",
        "deepseek",
    )
