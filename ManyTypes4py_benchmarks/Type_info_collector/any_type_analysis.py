import json
import re
import csv
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any as TypingAny


class AnyTypeAnalyzer:
    def __init__(self):
        self.any_patterns = [
            r"\bAny\b",
            # r"\bList\[Any\]",
            # r"\bDict\[.*?Any.*?\]",
            # r"\bTuple\[.*?Any.*?\]",
            # r"\bUnion\[.*?Any.*?\]",
            # r"\bOptional\[Any\]",
            # r"\bSet\[Any\]",
            # r"\btyping\.Any\b",
            # r"\bList\[List\[Any\]\]",
            # r"\bDict\[Any, Any\]",
            # r"\bDict\[str, Any\]",
            # r"\bDict\[Any, str\]",
        ]
        self.any_regex = re.compile("|".join(self.any_patterns), re.IGNORECASE)

    def load_json_data(self, filepath: str) -> Dict:
        """Load JSON data from file."""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}

    def load_coverage_bins(self, filepath: str) -> Dict[str, List[str]]:
        """Load coverage bin data."""
        return self.load_json_data(filepath)

    def extract_any_types(self, type_annotation: str) -> List[str]:
        """Extract all Any-related types from a type annotation."""
        if not type_annotation:
            return []

        matches = self.any_regex.findall(type_annotation)
        return [match for match in matches if match]

    def analyze_file_types(self, file_data: Dict) -> Dict:
        """Analyze types in a single file."""
        any_counts = {
            "total_annotations": 0,
            "any_annotations": 0,
            "any_types": [],
            "function_params": {"total": 0, "any": 0},
            "function_returns": {"total": 0, "any": 0},
            "variables": {"total": 0, "any": 0},
            "class_attributes": {"total": 0, "any": 0},
        }

        # Process each function/class in the file
        for func_name, annotations in file_data.items():
            for annotation in annotations:
                category = annotation.get("category", "")
                name = annotation.get("name", "")
                type_list = annotation.get("type", [])

                # Process each type in the type list
                for type_annotation in type_list:
                    if type_annotation:  # Skip empty type annotations
                        any_counts["total_annotations"] += 1

                        # Categorize by annotation type
                        if category == "arg":
                            any_counts["function_params"]["total"] += 1
                        elif category == "return":
                            any_counts["function_returns"]["total"] += 1
                        elif category == "var":
                            any_counts["variables"]["total"] += 1
                        elif category == "attr":
                            any_counts["class_attributes"]["total"] += 1

                        # Check for Any types
                        any_types = self.extract_any_types(type_annotation)
                        if any_types:
                            any_counts["any_annotations"] += 1
                            any_counts["any_types"].extend(any_types)

                            # Update category-specific counts
                            if category == "arg":
                                any_counts["function_params"]["any"] += 1
                            elif category == "return":
                                any_counts["function_returns"]["any"] += 1
                            elif category == "var":
                                any_counts["variables"]["any"] += 1
                            elif category == "attr":
                                any_counts["class_attributes"]["any"] += 1

        return any_counts

    def analyze_dataset(
        self, type_info_file: str, coverage_bins_file: str, dataset_name: str
    ) -> Dict:
        """Analyze entire dataset."""
        print(f"Loading {dataset_name} data...")
        type_data = self.load_json_data(type_info_file)
        coverage_bins = self.load_coverage_bins(coverage_bins_file)

        # Overall statistics
        total_files = len(type_data)
        total_annotations = 0
        total_any_annotations = 0
        any_type_counter = Counter()

        # Per-file analysis
        file_analyses = {}

        for filename, file_data in type_data.items():
            file_analysis = self.analyze_file_types(file_data)
            file_analyses[filename] = file_analysis

            total_annotations += file_analysis["total_annotations"]
            total_any_annotations += file_analysis["any_annotations"]
            any_type_counter.update(file_analysis["any_types"])

        # Coverage bin analysis
        bin_analysis = defaultdict(
            lambda: {
                "files": 0,
                "total_annotations": 0,
                "any_annotations": 0,
                "any_percentage": 0.0,
            }
        )

        for bin_name, files_in_bin in coverage_bins.items():
            for filename in files_in_bin:
                if filename in file_analyses:
                    analysis = file_analyses[filename]
                    bin_analysis[bin_name]["files"] += 1
                    bin_analysis[bin_name]["total_annotations"] += analysis[
                        "total_annotations"
                    ]
                    bin_analysis[bin_name]["any_annotations"] += analysis[
                        "any_annotations"
                    ]

        # Calculate percentages
        for bin_name in bin_analysis:
            total = bin_analysis[bin_name]["total_annotations"]
            if total > 0:
                bin_analysis[bin_name]["any_percentage"] = (
                    bin_analysis[bin_name]["any_annotations"] / total * 100
                )

        return {
            "dataset_name": dataset_name,
            "total_files": total_files,
            "total_annotations": total_annotations,
            "total_any_annotations": total_any_annotations,
            "overall_any_percentage": (
                (total_any_annotations / total_annotations * 100)
                if total_annotations > 0
                else 0
            ),
            "any_type_distribution": dict(any_type_counter),
            "file_analyses": file_analyses,
            "bin_analysis": dict(bin_analysis),
        }

    def compare_all_datasets(self, analyses: Dict[str, Dict]) -> Dict:
        """Compare all datasets (Human, O1-Mini, DeepSeek, GPT4o)."""
        comparison = {
            "overall_comparison": {},
            "bin_comparison": {},
            "any_type_comparison": {},
        }

        # Overall comparison
        for dataset_name, analysis in analyses.items():
            comparison["overall_comparison"][dataset_name] = {
                "any_percentage": analysis["overall_any_percentage"],
                "total_annotations": analysis["total_annotations"],
                "any_annotations": analysis["total_any_annotations"],
            }

        # Compare by coverage bins
        all_bins = set()
        for analysis in analyses.values():
            all_bins.update(analysis["bin_analysis"].keys())

        for bin_name in sorted(all_bins):
            bin_data = {}
            for dataset_name, analysis in analyses.items():
                bin_info = analysis["bin_analysis"].get(bin_name, {})
                bin_data[dataset_name] = {
                    "any_percentage": bin_info.get("any_percentage", 0),
                    "files": bin_info.get("files", 0),
                    "total_annotations": bin_info.get("total_annotations", 0),
                    "any_annotations": bin_info.get("any_annotations", 0),
                }
            comparison["bin_comparison"][bin_name] = bin_data

        # Compare any type distributions
        all_types = set()
        for analysis in analyses.values():
            all_types.update(analysis["any_type_distribution"].keys())

        for type_name in all_types:
            type_data = {}
            for dataset_name, analysis in analyses.items():
                type_data[dataset_name] = analysis["any_type_distribution"].get(
                    type_name, 0
                )
            comparison["any_type_comparison"][type_name] = type_data

        return comparison

    def save_coverage_bin_csv(
        self, comparison: Dict, filename: str = "coverage_bin_comparison_only_any.csv"
    ):
        """Save coverage bin comparison results to CSV file."""
        with open(filename, "w", newline="") as csvfile:
            fieldnames = [
                "Bin",
                "Human_Any_Pct",
                "Human_Files",
                "O1_Mini_Any_Pct",
                "O1_Mini_Files",
                "DeepSeek_Any_Pct",
                "DeepSeek_Files",
                "GPT4o_Any_Pct",
                "GPT4o_Files",
            ]

            # Add O3-Mini and Claude3-Sonnet if they exist
            if "O3-Mini" in comparison["overall_comparison"]:
                fieldnames.extend(["O3_Mini_Any_Pct", "O3_Mini_Files"])
            if "Claude3-Sonnet" in comparison["overall_comparison"]:
                fieldnames.extend(["Claude3_Sonnet_Any_Pct", "Claude3_Sonnet_Files"])

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Sort bins by key values (handle numeric ranges properly)
            def bin_sort_key(bin_name):
                if bin_name == "100%":
                    return 100
                # Extract the first number from ranges like "0-5%", "5-10%", etc.
                return int(bin_name.split("-")[0])

            sorted_bins = sorted(
                comparison["bin_comparison"].items(), key=lambda x: bin_sort_key(x[0])
            )

            for bin_name, bin_data in sorted_bins:
                row = {"Bin": bin_name}

                # Add data for each LLM
                for dataset_name, data in bin_data.items():
                    if data["files"] > 0:
                        row[f'{dataset_name.replace("-", "_")}_Any_Pct'] = (
                            f"{data['any_percentage']:.2f}"
                        )
                        row[f'{dataset_name.replace("-", "_")}_Files'] = data["files"]
                    else:
                        row[f'{dataset_name.replace("-", "_")}_Any_Pct'] = ""
                        row[f'{dataset_name.replace("-", "_")}_Files'] = ""

                writer.writerow(row)

        print(f"\nCoverage bin comparison saved to {filename}")

    def print_results(self, analysis: Dict):
        """Print analysis results."""
        print(f"\n=== {analysis['dataset_name']} Analysis ===")
        print(f"Total files: {analysis['total_files']}")
        print(f"Total annotations: {analysis['total_annotations']}")
        print(f"Any annotations: {analysis['total_any_annotations']}")
        print(f"Overall Any percentage: {analysis['overall_any_percentage']:.2f}%")

        print("\nTop Any types:")
        for type_name, count in sorted(
            analysis["any_type_distribution"].items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {type_name}: {count}")

        print("\nCoverage bin analysis:")

        # Sort bins by key values (handle numeric ranges properly)
        def bin_sort_key(bin_name):
            if bin_name == "100%":
                return 100
            # Extract the first number from ranges like "0-5%", "5-10%", etc.
            return int(bin_name.split("-")[0])

        sorted_bins = sorted(
            analysis["bin_analysis"].items(), key=lambda x: bin_sort_key(x[0])
        )
        for bin_name, bin_data in sorted_bins:
            if bin_data["files"] > 0:
                print(
                    f"  {bin_name}: {bin_data['any_percentage']:.2f}% ({bin_data['files']} files)"
                )


def main():
    analyzer = AnyTypeAnalyzer()

    # File paths for all LLMs
    datasets = {
        "Human": {
            "type_file": "Type_info_original_files.json",
            "coverage_file": "../mypy_results/type_coverage_bins/coverage_bin_files_human.json",
        },
        "O1-Mini": {
            "type_file": "Type_info_o1_mini_benchmarks.json",
            "coverage_file": "../mypy_results/type_coverage_bins/coverage_bin_files_o1_mini.json",
        },
        "DeepSeek": {
            "type_file": "Type_info_deep_seek_benchmarks.json",
            "coverage_file": "../mypy_results/type_coverage_bins/coverage_bin_files_deepseek.json",
        },
        "GPT4o": {
            "type_file": "Type_info_gpt4o_benchmarks.json",
            "coverage_file": "../mypy_results/type_coverage_bins/coverage_bin_files_gpt4o.json",
        },
        "O3-Mini": {
            "type_file": "Type_info_o3_mini_1st_run_benchmarks.json",
            "coverage_file": "../mypy_results/type_coverage_bins/coverage_bin_files_o3_mini_1st_run.json",
        },
        "Claude3-Sonnet": {
            "type_file": "Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "coverage_file": "../mypy_results/type_coverage_bins/coverage_bin_files_claude_3_7_sonnet.json",
        },
    }

    # Analyze all datasets
    print("Starting analysis...")
    analyses = {}

    for dataset_name, file_paths in datasets.items():
        print(f"Analyzing {dataset_name}...")
        analysis = analyzer.analyze_dataset(
            file_paths["type_file"], file_paths["coverage_file"], dataset_name
        )
        analyses[dataset_name] = analysis
        analyzer.print_results(analysis)

    # Compare all datasets
    comparison = analyzer.compare_all_datasets(analyses)

    print("\n=== Overall Comparison Results ===")
    for dataset_name, data in comparison["overall_comparison"].items():
        print(
            f"{dataset_name}: {data['any_percentage']:.2f}% ({data['any_annotations']:,} out of {data['total_annotations']:,})"
        )

    print("\n=== Coverage Bin Comparison ===")

    # Sort bins by key values (handle numeric ranges properly)
    def bin_sort_key(bin_name):
        if bin_name == "100%":
            return 100
        # Extract the first number from ranges like "0-5%", "5-10%", etc.
        return int(bin_name.split("-")[0])

    sorted_bins = sorted(
        comparison["bin_comparison"].items(), key=lambda x: bin_sort_key(x[0])
    )
    for bin_name, bin_data in sorted_bins:
        print(f"\n{bin_name}:")
        for dataset_name, data in bin_data.items():
            if data["files"] > 0:
                print(
                    f"  {dataset_name}: {data['any_percentage']:.2f}% ({data['files']} files)"
                )

    # Save results
    results = {
        "analyses": analyses,
        "comparison": comparison,
    }

    # with open("any_analysis_results.json", "w") as f:
    #    json.dump(results, f, indent=2)

    print("\nResults saved to any_analysis_results.json")

    # Save coverage bin comparison to CSV
    analyzer.save_coverage_bin_csv(comparison)


if __name__ == "__main__":
    main()
