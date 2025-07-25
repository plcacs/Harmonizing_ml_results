import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def load_json_data(filepath: str) -> List[Dict]:
    """Load JSON data from file (expects a list of dicts)"""
    with open(filepath, "r") as f:
        return json.load(f)


def count_matches(data: List[Dict]) -> Tuple[int, int, int, int]:
    """Count different types of matches"""
    total = 0
    string_matches = 0
    semantic_matches = 0
    semantic_only_matches = 0

    for param in data:
        total += 1
        if param.get("exact_match", False):
            string_matches += 1
        if param.get("semantic_match", False):
            semantic_matches += 1
        if param.get("semantic_match", False) and not param.get("exact_match", False):
            semantic_only_matches += 1

    return total, string_matches, semantic_matches, semantic_only_matches


def analyze_file_level_performance(data: List[Dict]) -> Dict:
    """Analyze performance by file"""
    file_stats = {}
    file_groups = defaultdict(list)
    for param in data:
        file_groups[param.get("filename", "unknown")].append(param)

    for filename, params in file_groups.items():
        total = len(params)
        string_matches = sum(1 for p in params if p.get("exact_match", False))
        semantic_matches = sum(1 for p in params if p.get("semantic_match", False))
        semantic_only_matches = sum(1 for p in params if p.get("semantic_match", False) and not p.get("exact_match", False))
        if total > 0:
            file_stats[filename] = {
                "total": total,
                "string_matches": string_matches,
                "semantic_matches": semantic_matches,
                "semantic_only_matches": semantic_only_matches,
                "string_rate": string_matches / total * 100,
                "semantic_rate": semantic_matches / total * 100,
                "improvement": (semantic_matches - string_matches) / total * 100,
            }
    return file_stats


def analyze_type_complexity(type_str: str) -> str:
    """Analyze type complexity"""
    if not type_str or type_str.strip() == "":
        return "empty"
    type_str = type_str.strip("'\"")
    if "Union" in type_str or " | " in type_str:
        return "union"
    elif "Optional" in type_str:
        return "optional"
    elif (
        "List" in type_str
        or "Dict" in type_str
        or "Set" in type_str
        or "Tuple" in type_str
    ):
        return "generic"
    elif "[" in type_str and "]" in type_str:
        return "subscript"
    else:
        return "simple"


def analyze_errors(data: List[Dict]) -> Tuple[Dict, Dict]:
    """Analyze error patterns"""
    error_patterns = defaultdict(int)
    semantic_errors = defaultdict(int)
    for param in data:
        # Use Run_1 as human, Run_2 as LLM (or swap as needed)
        human_type = param.get("Run_1", "")
        llm_type = param.get("Run_2", "")
        if not param.get("exact_match", False):
            error_patterns[f"String mismatch: {human_type} vs {llm_type}"] += 1
        if not param.get("semantic_match", False):
            semantic_errors[f"Semantic mismatch: {human_type} vs {llm_type}"] += 1
    return dict(error_patterns), dict(semantic_errors)


def analyze_by_category(data: List[Dict]) -> Dict:
    """Analyze performance by category"""
    category_stats = defaultdict(
        lambda: {"total": 0, "string_matches": 0, "semantic_matches": 0}
    )
    for param in data:
        category = param.get("category", "unknown")
        category_stats[category]["total"] += 1
        if param.get("exact_match", False):
            category_stats[category]["string_matches"] += 1
        if param.get("semantic_match", False):
            category_stats[category]["semantic_matches"] += 1
    return dict(category_stats)


def analyze_type_complexity_performance(data: List[Dict]) -> Dict:
    """Analyze performance by type complexity"""
    complexity_stats = defaultdict(
        lambda: {"total": 0, "string_matches": 0, "semantic_matches": 0}
    )
    for param in data:
        human_type = param.get("Run_1", "")
        complexity = analyze_type_complexity(human_type)
        complexity_stats[complexity]["total"] += 1
        if param.get("exact_match", False):
            complexity_stats[complexity]["string_matches"] += 1
        if param.get("semantic_match", False):
            complexity_stats[complexity]["semantic_matches"] += 1
    return dict(complexity_stats)


def print_llm_performance_assessment(data: List[Dict], llm_name: str, output_file):
    """Print LLM Performance Assessment"""
    total, string_matches, semantic_matches, semantic_only = count_matches(data)

    output = f"\n{'='*60}\n"
    output += f"LLM PERFORMANCE ASSESSMENT - {llm_name.upper()}\n"
    output += f"{'='*60}\n"

    output += f"Total type comparisons: {total}\n"
    output += f"String matches: {string_matches} ({string_matches/total*100:.2f}%)\n"
    output += (
        f"Semantic matches: {semantic_matches} ({semantic_matches/total*100:.2f}%)\n"
    )
    output += (
        f"Semantic-only matches: {semantic_only} ({semantic_only/total*100:.2f}%)\n"
    )
    output += f"Improvement with semantic matching: {semantic_matches - string_matches} ({((semantic_matches - string_matches)/total)*100:.2f}%)\n"

    print(output)
    output_file.write(output)


def print_file_level_analysis(file_stats: Dict, output_file):
    """Print file-level analysis"""
    output = f"\n{'='*60}\n"
    output += "FILE-LEVEL ANALYSIS (Top 20 files by improvement)\n"
    output += f"{'='*60}\n"

    # Sort by improvement percentage
    sorted_files = sorted(
        file_stats.items(), key=lambda x: x[1]["improvement"], reverse=True
    )

    output += f"{'Filename':<50} {'Total':<6} {'String%':<8} {'Semantic%':<10} {'Improvement%':<12}\n"
    output += f"{'-'*50} {'-'*6} {'-'*8} {'-'*10} {'-'*12}\n"

    for filename, stats in sorted_files[:20]:
        output += f"{filename[:49]:<50} {stats['total']:<6} {stats['string_rate']:<8.2f} {stats['semantic_rate']:<10.2f} {stats['improvement']:<12.2f}\n"

    output += f"\n{'='*60}\n"
    output += "FILE-LEVEL ANALYSIS (Bottom 20 files by improvement)\n"
    output += f"{'='*60}\n"

    output += f"{'Filename':<50} {'Total':<6} {'String%':<8} {'Semantic%':<10} {'Improvement%':<12}\n"
    output += f"{'-'*50} {'-'*6} {'-'*8} {'-'*10} {'-'*12}\n"

    for filename, stats in sorted_files[-20:]:
        output += f"{filename[:49]:<50} {stats['total']:<6} {stats['string_rate']:<8.2f} {stats['semantic_rate']:<10.2f} {stats['improvement']:<12.2f}\n"

    print(output)
    output_file.write(output)


def print_category_analysis(category_stats: Dict, output_file):
    """Print category-wise analysis"""
    output = f"\n{'='*60}\n"
    output += "CATEGORY-WISE ANALYSIS\n"
    output += f"{'='*60}\n"

    for category, stats in category_stats.items():
        total = stats["total"]
        string_rate = stats["string_matches"] / total * 100 if total > 0 else 0
        semantic_rate = stats["semantic_matches"] / total * 100 if total > 0 else 0

        output += f"{category.upper():<15}: {total:>5} total | String: {string_rate:>6.2f}% | Semantic: {semantic_rate:>6.2f}% | Improvement: {semantic_rate - string_rate:>6.2f}%\n"

    print(output)
    output_file.write(output)


def print_complexity_analysis(complexity_stats: Dict, output_file):
    """Print type complexity analysis"""
    output = f"\n{'='*60}\n"
    output += "TYPE COMPLEXITY ANALYSIS\n"
    output += f"{'='*60}\n"

    for complexity, stats in complexity_stats.items():
        total = stats["total"]
        string_rate = stats["string_matches"] / total * 100 if total > 0 else 0
        semantic_rate = stats["semantic_matches"] / total * 100 if total > 0 else 0

        output += f"{complexity.upper():<15}: {total:>5} total | String: {string_rate:>6.2f}% | Semantic: {semantic_rate:>6.2f}% | Improvement: {semantic_rate - string_rate:>6.2f}%\n"

    print(output)
    output_file.write(output)


def print_error_analysis(error_patterns: Dict, semantic_errors: Dict, output_file):
    """Print error analysis"""
    output = f"\n{'='*60}\n"
    output += "TOP 10 STRING MATCH ERRORS\n"
    output += f"{'='*60}\n"

    for error, count in sorted(
        error_patterns.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        output += f"{count:>4}: {error}\n"

    output += f"\n{'='*60}\n"
    output += "TOP 10 SEMANTIC MATCH ERRORS\n"
    output += f"{'='*60}\n"

    for error, count in sorted(
        semantic_errors.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        output += f"{count:>4}: {error}\n"

    print(output)
    output_file.write(output)


def main():
    """Main analysis function"""
    # Get all semantic comparison files
    json_files = [
        f
        for f in os.listdir(".")
        if f.startswith("llm_vs_llm_comparison_") and f.endswith(".json")
    ]

    if not json_files:
        print("No semantic comparison JSON files found!")
        return

    print(f"Found {len(json_files)} semantic comparison files")

    for json_file in json_files:
        llm_name = json_file.replace("llm_vs_llm_comparison_", "").replace(
            ".json", ""
        )
        output_filename = f"analysis_results_{llm_name}.txt"

        print(f"\n{'='*80}")
        print(f"ANALYZING: {json_file}")
        print(f"OUTPUT: {output_filename}")
        print(f"{'='*80}")

        try:
            data = load_json_data(json_file)

            with open(output_filename, "w") as output_file:
                output_file.write(
                    f"SEMANTIC TYPE COMPARISON ANALYSIS - {llm_name.upper()}\n"
                )
                output_file.write(f"Generated from: {json_file}\n")
                output_file.write(f"{'='*80}\n")

                # Perform analyses
                print_llm_performance_assessment(data, llm_name, output_file)

                file_stats = analyze_file_level_performance(data)
                print_file_level_analysis(file_stats, output_file)

                category_stats = analyze_by_category(data)
                print_category_analysis(category_stats, output_file)

                complexity_stats = analyze_type_complexity_performance(data)
                print_complexity_analysis(complexity_stats, output_file)

                error_patterns, semantic_errors = analyze_errors(data)
                print_error_analysis(error_patterns, semantic_errors, output_file)

            print(f"Analysis results saved to: {output_filename}")

        except Exception as e:
            print(f"Error analyzing {json_file}: {e}")


if __name__ == "__main__":
    main()
