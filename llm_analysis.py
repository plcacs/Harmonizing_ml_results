#!/usr/bin/env python3
"""
Generic LLM analysis script for mypy results, any-ratio, and type info data.
This script compares performance across original, partial, and full typing scenarios for any LLM.
"""

import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    llm_name: str
    total_files: int
    non_any_ratios: Tuple[float, float, float]  # O, U, H (compiled files only)
    non_any_ratios_all: Tuple[float, float, float]  # O, U, H (all files)
    precision_ratios: Tuple[float, float, float]  # O, U, H
    compiled_files: List[str]


class LLMAnalyzer:
    def __init__(self, llm_name: str):
        self.llm_name = llm_name.lower()
        self.base_path = "ManyTypes4py_benchmarks"
        self.setup_file_paths()

    def setup_file_paths(self):
        """Setup file paths based on LLM name."""
        if self.llm_name == "claude":
            # Claude-specific file paths based on actual directory structure
            self.mypy_files = {
                "partial": "mypy_results/mypy_outputs/partial_typed/mypy_results_claude3_sonnet_partially_typed_files_with_errors.json",
                "full": "mypy_results/mypy_outputs/mypy_results_claude3_sonnet_user_annotated_with_errors.json",
                "original": "mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
            }

            self.any_ratio_files = {
                "partial": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/Claude3-Sonnet_Partially_Typed/per_file_any_percentage.json",
                "full": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/Claude3-Sonnet_User_Annotated/per_file_any_percentage.json",
                "original": "Type_info_collector/Section_04/per_file_any_percentage/Claude3-Sonnet_1st_run/per_file_any_percentage.json",
            }

            self.typeinfo_files = {
                "partial": "Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_partially_typed_files_benchmarks.json",
                "full": "Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_user_annotated_benchmarks.json",
                "original": "Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            }
        elif self.llm_name == "o3-mini":
            self.mypy_files = {
                "partial": "mypy_results/mypy_outputs/partial_typed/mypy_results_o3_mini_partially_typed_files_with_errors.json",
                "full": "mypy_results/mypy_outputs/mypy_results_o3_mini_user_annotated_with_errors.json",
                "original": "mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
            }

            self.any_ratio_files = {
                "partial": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/O3-mini_Partially_Typed/per_file_any_percentage.json",
                "full": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/O3-mini_User_Annotated/per_file_any_percentage.json",
                "original": "Type_info_collector/Section_04/per_file_any_percentage/O3-mini_1st_run/per_file_any_percentage.json",
            }

            self.typeinfo_files = {
                "partial": "Type_info_collector/Type_info_LLMS/Type_info_o3_mini_partially_typed_files_benchmarks.json",
                "full": "Type_info_collector/Type_info_LLMS/Type_info_o3_mini_user_annotated_benchmarks.json",
                "original": "Type_info_collector/Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            }
        elif self.llm_name == "deepseek":
            self.mypy_files = {
                "partial": "mypy_results/mypy_outputs/partial_typed/mypy_results_deepseek_partially_typed_files_with_errors.json",
                "full": "mypy_results/mypy_outputs/mypy_results_deepseek_user_annotated_with_errors.json",
                "original": "mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
            }

            self.any_ratio_files = {
                "partial": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/DeepSeek_Partially_Typed/per_file_any_percentage.json",
                "full": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/DeepSeek_User_Annotated/per_file_any_percentage.json",
                "original": "Type_info_collector/Section_04/per_file_any_percentage/DeepSeek_1st_run/per_file_any_percentage.json",
            }

            self.typeinfo_files = {
                "partial": "Type_info_collector/Type_info_LLMS/Type_info_deepseek_partially_typed_files_benchmarks.json",
                "full": "Type_info_collector/Type_info_LLMS/Type_info_deepseek_user_annotated_benchmarks.json",
                "original": "Type_info_collector/Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            }

        else:
            raise ValueError(
                f"LLM '{self.llm_name}' not supported. Currently only 'claude' is supported."
            )

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse a JSON file."""
        full_path = os.path.join(self.base_path, file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found: {full_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {full_path}: {e}")
            return {}

    def find_common_compiled_files(self) -> List[str]:
        """Find files that have isCompiled: true in all three mypy result files."""
        print(f"Loading mypy results for {self.llm_name.upper()}...")

        mypy_data = {}
        for scenario, file_path in self.mypy_files.items():
            mypy_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(mypy_data[scenario])} files")

        # Find files that compile successfully in all scenarios
        all_files = (
            set(mypy_data["original"].keys())
            & set(mypy_data["partial"].keys())
            & set(mypy_data["full"].keys())
        )

        compiled_files = []
        for filename in all_files:
            if (
                mypy_data["original"][filename].get("isCompiled", False)
                and mypy_data["partial"][filename].get("isCompiled", False)
                and mypy_data["full"][filename].get("isCompiled", False)
            ):
                compiled_files.append(filename)

        print(f"Found {len(compiled_files)} files that compile in all scenarios")
        print(f"Total common files: {len(all_files)}")

        return compiled_files

    def calculate_non_any_ratios(
        self, compiled_files: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate ratios for which scenario has the highest non-Any ratio (filtered by compiled files)."""
        print("Loading any-ratio data...")

        any_ratio_data = {}
        for scenario, file_path in self.any_ratio_files.items():
            any_ratio_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(any_ratio_data[scenario])} files")

        # Initialize counters
        counters = {"O": 0, "U": 0, "H": 0}
        scenario_stats = defaultdict(list)

        print(
            f"Analyzing {len(compiled_files)} files for non-Any ratios (compiled files only)..."
        )

        tie_stats = {"no_tie": 0, "two_way_tie": 0, "three_way_tie": 0}

        for filename in compiled_files:
            # Get any percentages for each scenario
            original_any = (
                any_ratio_data["original"]
                .get(filename, {})
                .get("any_percentage", 100.0)
            )
            partial_any = (
                any_ratio_data["partial"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )
            full_any = (
                any_ratio_data["full"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )

            # Calculate non-any ratios (100 - any_percentage)
            ratios = {
                "U": 100 - original_any,
                "H": 100 - partial_any,
                "O": 100 - full_any,
            }

            scenario_stats["original"].append(ratios["U"])
            scenario_stats["partial"].append(ratios["H"])
            scenario_stats["full"].append(ratios["O"])

            # Find the highest non-any ratio
            max_ratio = max(ratios.values())

            # Count how many scenarios tie for the highest ratio
            winners = [
                scenario for scenario, ratio in ratios.items() if ratio == max_ratio
            ]
            if len(winners) == 1:
                tie_stats["no_tie"] += 1
                # Only count when there's a clear winner (no ties)
                counters[winners[0]] += 1
            elif len(winners) == 2:
                tie_stats["two_way_tie"] += 1
                # Don't increment counters for ties
            else:
                tie_stats["three_way_tie"] += 1
                # Don't increment counters for ties

        # Calculate ratios
        total_files = len(compiled_files)
        ratios = (
            counters["O"] / total_files,
            counters["U"] / total_files,
            counters["H"] / total_files,
        )

        # Print detailed statistics
        print(f"Non-Any Ratio Statistics (compiled files only):")
        for scenario, values in scenario_stats.items():
            avg_ratio = sum(values) / len(values)
            print(f"  {scenario}: avg non-any ratio = {avg_ratio:.2f}%")

        print(f"Tie Analysis:")
        print(f"  No ties: {tie_stats['no_tie']} files")
        print(f"  Two-way ties: {tie_stats['two_way_tie']} files")
        print(f"  Three-way ties: {tie_stats['three_way_tie']} files")

        # Store file counts for display
        self.non_any_file_counts = {
            "O": counters["O"],
            "U": counters["U"],
            "H": counters["H"],
        }

        return ratios

    def calculate_non_any_ratios_all_files(self) -> Tuple[float, float, float]:
        """Calculate ratios for which scenario has the highest non-Any ratio (all files, ignoring compilation status)."""
        print("Loading any-ratio data for all files...")

        any_ratio_data = {}
        for scenario, file_path in self.any_ratio_files.items():
            any_ratio_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(any_ratio_data[scenario])} files")

        # Find all files that exist in at least one scenario
        all_files = set()
        for scenario_data in any_ratio_data.values():
            all_files.update(scenario_data.keys())

        print(f"Found {len(all_files)} total files across all scenarios")

        # Initialize counters
        counters = {"O": 0, "U": 0, "H": 0}
        scenario_stats = defaultdict(list)

        print(f"Analyzing {len(all_files)} files for non-Any ratios (all files)...")

        for filename in all_files:
            # Get any percentages for each scenario
            original_any = (
                any_ratio_data["original"]
                .get(filename, {})
                .get("any_percentage", 100.0)
            )
            partial_any = (
                any_ratio_data["partial"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )
            full_any = (
                any_ratio_data["full"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )

            # Calculate non-any ratios (100 - any_percentage)
            ratios = {
                "U": 100 - original_any,
                "H": 100 - partial_any,
                "O": 100 - full_any,
            }

            scenario_stats["original"].append(ratios["U"])
            scenario_stats["partial"].append(ratios["H"])
            scenario_stats["full"].append(ratios["O"])

            # Find the highest non-any ratio
            max_ratio = max(ratios.values())

            # Count which scenarios have the highest ratio (only when there's a clear winner)
            winners = [
                scenario for scenario, ratio in ratios.items() if ratio == max_ratio
            ]
            if len(winners) == 1:
                counters[winners[0]] += 1

        # Calculate ratios
        total_files = len(all_files)
        ratios = (
            counters["O"] / total_files,
            counters["U"] / total_files,
            counters["H"] / total_files,
        )

        # Print detailed statistics
        print(f"Non-Any Ratio Statistics (all files):")
        for scenario, values in scenario_stats.items():
            avg_ratio = sum(values) / len(values)
            print(f"  {scenario}: avg non-any ratio = {avg_ratio:.2f}%")

        # Store file counts for display
        self.non_any_file_counts_all = {
            "O": counters["O"],
            "U": counters["U"],
            "H": counters["H"],
        }

        return ratios

    def calculate_precision_ratios(
        self, compiled_files: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate ratios for which scenario is more precise using typeinfo data."""
        print("Loading typeinfo data...")

        typeinfo_data = {}
        for scenario, file_path in self.typeinfo_files.items():
            typeinfo_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(typeinfo_data[scenario])} files")

        # Initialize counters
        counters = {"O": 0, "U": 0, "H": 0}
        precision_stats = defaultdict(list)

        print(f"Analyzing {len(compiled_files)} files for precision...")

        for filename in compiled_files:
            original_info = typeinfo_data["original"].get(filename, {})
            partial_info = typeinfo_data["partial"].get(filename, original_info)
            full_info = typeinfo_data["full"].get(filename, original_info)

            # Calculate precision metrics
            precisions = {
                "U": self.calculate_file_precision(original_info),
                "H": self.calculate_file_precision(partial_info),
                "O": self.calculate_file_precision(full_info),
            }

            precision_stats["original"].append(precisions["U"])
            precision_stats["partial"].append(precisions["H"])
            precision_stats["full"].append(precisions["O"])

            # Find the highest precision
            max_precision = max(precisions.values())

            # Count which scenarios have the highest precision (only when there's a clear winner)
            winners = [
                scenario
                for scenario, precision in precisions.items()
                if precision == max_precision
            ]
            if len(winners) == 1:
                counters[winners[0]] += 1

        # Calculate ratios
        total_files = len(compiled_files)
        ratios = (
            counters["O"] / total_files,
            counters["U"] / total_files,
            counters["H"] / total_files,
        )

        # Print detailed statistics
        print(f"Precision Statistics:")
        for scenario, values in precision_stats.items():
            avg_precision = sum(values) / len(values)
            print(f"  {scenario}: avg precision = {avg_precision:.2f}%")

        # Store file counts for display
        self.precision_file_counts = {
            "O": counters["O"],
            "U": counters["U"],
            "H": counters["H"],
        }

        return ratios

    def calculate_file_precision(self, file_info: Dict) -> float:
        """Calculate precision score for a file based on typeinfo data."""
        if not file_info:
            return 0.0

        total_types = 0
        specific_types = 0

        for function_name, type_list in file_info.items():
            for type_info in type_list:
                if type_info.get("category") in ["arg", "return"]:
                    types = type_info.get("type", [])
                    for type_val in types:
                        total_types += 1
                        # Count as specific if it's not empty and not 'Any'
                        if type_val and type_val.strip() and type_val != "Any":
                            specific_types += 1

        if total_types == 0:
            return 0.0

        return (specific_types / total_types) * 100

    def run_analysis(self) -> AnalysisResult:
        """Run the complete analysis."""
        print(f"=== {self.llm_name.upper()} Analysis ===\n")

        # Step 1: Find common compiled files
        compiled_files = self.find_common_compiled_files()
        if not compiled_files:
            print("No common compiled files found!")
            return None

        print(f"Analyzing {len(compiled_files)} files\n")

        # Step 2: Non-Any ratio analysis (compiled files only)
        print("=== Non-Any Ratio Analysis (Compiled Files Only) ===")
        non_any_ratios = self.calculate_non_any_ratios(compiled_files)
        O_ratio, U_ratio, H_ratio = non_any_ratios

        print(f"Results:")
        print(
            f"  Full (O) ratio: {O_ratio:.4f} ({O_ratio*100:.2f}%) - {self.non_any_file_counts['O']} files"
        )
        print(
            f"  Original (U) ratio: {U_ratio:.4f} ({U_ratio*100:.2f}%) - {self.non_any_file_counts['U']} files"
        )
        print(
            f"  Partial (H) ratio: {H_ratio:.4f} ({H_ratio*100:.2f}%) - {self.non_any_file_counts['H']} files"
        )
        print()

        # Step 2.5: Non-Any ratio analysis (all files)
        print("=== Non-Any Ratio Analysis (All Files) ===")
        non_any_ratios_all = self.calculate_non_any_ratios_all_files()
        O_ratio_all, U_ratio_all, H_ratio_all = non_any_ratios_all

        print(f"Results:")
        print(
            f"  Full (O) ratio: {O_ratio_all:.4f} ({O_ratio_all*100:.2f}%) - {self.non_any_file_counts_all['O']} files"
        )
        print(
            f"  Original (U) ratio: {U_ratio_all:.4f} ({U_ratio_all*100:.2f}%) - {self.non_any_file_counts_all['U']} files"
        )
        print(
            f"  Partial (H) ratio: {H_ratio_all:.4f} ({H_ratio_all*100:.2f}%) - {self.non_any_file_counts_all['H']} files"
        )
        print()

        # Step 3: Precision analysis
        print("=== Precision Analysis ===")
        precision_ratios = self.calculate_precision_ratios(compiled_files)
        O_prec, U_prec, H_prec = precision_ratios

        print(f"Results:")
        print(
            f"  Full (O) precision ratio: {O_prec:.4f} ({O_prec*100:.2f}%) - {self.precision_file_counts['O']} files"
        )
        print(
            f"  Original (U) precision ratio: {U_prec:.4f} ({U_prec*100:.2f}%) - {self.precision_file_counts['U']} files"
        )
        print(
            f"  Partial (H) precision ratio: {H_prec:.4f} ({H_prec*100:.2f}%) - {self.precision_file_counts['H']} files"
        )
        print()

        # Summary
        print("=== Summary ===")
        print(f"Total compiled files analyzed: {len(compiled_files)}")

        non_any_winner = (
            "Full"
            if O_ratio > max(U_ratio, H_ratio)
            else "Partial" if H_ratio > U_ratio else "Original"
        )
        non_any_winner_all = (
            "Full"
            if O_ratio_all > max(U_ratio_all, H_ratio_all)
            else "Partial" if H_ratio_all > U_ratio_all else "Original"
        )
        precision_winner = (
            "Full"
            if O_prec > max(U_prec, H_prec)
            else "Partial" if H_prec > U_prec else "Original"
        )

        print(f"Non-Any Ratio Winner (compiled files): {non_any_winner}")
        print(f"Non-Any Ratio Winner (all files): {non_any_winner_all}")
        print(f"Precision Winner: {precision_winner}")

        return AnalysisResult(
            llm_name=self.llm_name,
            total_files=len(compiled_files),
            non_any_ratios=non_any_ratios,
            non_any_ratios_all=non_any_ratios_all,
            precision_ratios=precision_ratios,
            compiled_files=compiled_files,
        )


def analyze_llm(llm_name: str) -> AnalysisResult:
    """Analyze a specific LLM and return results."""
    analyzer = LLMAnalyzer(llm_name)
    return analyzer.run_analysis()


def compare_llms(llm_names: List[str]) -> Dict[str, AnalysisResult]:
    """Compare multiple LLMs."""
    results = {}
    for llm_name in llm_names:
        print(f"\n{'='*60}")
        results[llm_name] = analyze_llm(llm_name)
        print(f"{'='*60}")

    return results


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_analysis.py claude")
        print("Currently only Claude is supported.")
        print("Example: python llm_analysis.py claude")
        return

    llm_names = sys.argv[1:]

    if len(llm_names) == 1:
        # Single LLM analysis
        result = analyze_llm(llm_names[0])
        if result:
            print(f"\n=== Final Results for {result.llm_name.upper()} ===")
            print(f"Compiled files analyzed: {result.total_files}")
            print(f"Non-Any ratios (compiled files) (O, U, H): {result.non_any_ratios}")
            print(f"Non-Any ratios (all files) (O, U, H): {result.non_any_ratios_all}")
            print(f"Precision ratios (O, U, H): {result.precision_ratios}")
    else:
        # Multiple LLM comparison
        results = compare_llms(llm_names)

        print(f"\n{'='*80}")
        print("=== COMPARISON SUMMARY ===")
        print(f"{'='*80}")

        for llm_name, result in results.items():
            if result:
                print(f"\n{llm_name.upper()}:")
                print(f"  Compiled files: {result.total_files}")
                print(
                    f"  Non-Any ratios (compiled) (O, U, H): {[f'{r:.3f}' for r in result.non_any_ratios]}"
                )
                print(
                    f"  Non-Any ratios (all files) (O, U, H): {[f'{r:.3f}' for r in result.non_any_ratios_all]}"
                )
                print(
                    f"  Precision ratios (O, U, H): {[f'{r:.3f}' for r in result.precision_ratios]}"
                )


if __name__ == "__main__":
    main()
