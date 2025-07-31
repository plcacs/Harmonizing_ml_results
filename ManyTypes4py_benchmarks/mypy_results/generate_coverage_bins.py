#!/usr/bin/env python3
"""
Type Coverage Bins Generator

This script generates type coverage bins JSON files from mypy results.
It organizes files into coverage percentage bins (0-5%, 5-10%, etc.)
and saves them in the same format as existing coverage_bin_files_*.json files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


class CoverageBinsGenerator:
    def __init__(self, mypy_results_dir: str):
        self.mypy_results_dir = Path(mypy_results_dir)

    def load_mypy_results(self, model_name: str, file_path: str) -> Dict:
        """Load mypy results for a specific model."""
        full_path = self.mypy_results_dir / file_path
        if not full_path.exists():
            print(f"Warning: File {full_path} not found")
            return {}

        with open(full_path, "r") as f:
            return json.load(f)

    def calculate_file_coverage(self, mypy_data: Dict) -> Dict[str, float]:
        """Calculate type coverage for each file."""
        file_coverage = {}

        for filename, file_data in mypy_data.items():
            stats = file_data.get("stats", {})
            total_params = stats.get("total_parameters", 0)
            annotated_params = stats.get("parameters_with_annotations", 0)

            if total_params > 0:
                coverage = (annotated_params / total_params) * 100
                file_coverage[filename] = coverage
            else:
                # If no parameters, set coverage to 0
                file_coverage[filename] = 0.0

        return file_coverage

    def create_coverage_bins(
        self, file_coverage: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Create coverage bins from file coverage data."""
        bins = defaultdict(list)

        # Define coverage ranges
        coverage_ranges = [
            (0, 5, "0-5%"),
            (5, 10, "5-10%"),
            (10, 20, "10-20%"),
            (20, 30, "20-30%"),
            (30, 40, "30-40%"),
            (40, 50, "40-50%"),
            (50, 60, "50-60%"),
            (60, 70, "60-70%"),
            (70, 80, "70-80%"),
            (80, 90, "80-90%"),
            (90, 100, "90-100%"),
            (100, 101, "100%"),  # Exactly 100%
        ]

        for filename, coverage in file_coverage.items():
            for min_coverage, max_coverage, bin_name in coverage_ranges:
                if min_coverage <= coverage < max_coverage:
                    bins[bin_name].append(filename)
                    break

        # Convert defaultdict to regular dict and sort files in each bin
        result = {}
        for bin_name in sorted(
            bins.keys(),
            key=lambda x: (
                float(x.split("-")[0]) if "-" in x else float(x.replace("%", ""))
            ),
        ):
            result[bin_name] = sorted(bins[bin_name])

        return result

    def generate_all_models(self):
        """Generate coverage bins for all models."""
        model_configs = [
            ("Human", "mypy_results_original_files_with_errors.json"),
            ("o3_mini_1st_run", "mypy_results_o3_mini_1st_run_with_errors.json"),
            (
                "claude_3_7_sonnet",
                "mypy_results_claude3_sonnet_1st_run_with_errors.json",
            ),
            ("gpt4o", "mypy_results_gpt4o_with_errors.json"),
            ("o1-mini", "mypy_results_o1_mini_with_errors.json"),
            ("deepseek", "mypy_results_deepseek_with_errors.json"),
        ]

        output_dir = Path("type_coverage_bins")
        output_dir.mkdir(exist_ok=True)

        for model_name, file_path in model_configs:
            print(f"Processing {model_name}...")

            # Load mypy results
            mypy_data = self.load_mypy_results(model_name, file_path)
            if not mypy_data:
                print(f"Warning: No data found for {model_name}")
                continue

            # Calculate file coverage
            file_coverage = self.calculate_file_coverage(mypy_data)

            # Create coverage bins
            coverage_bins = self.create_coverage_bins(file_coverage)

            # Save to JSON file
            output_filename = f"coverage_bin_files_{model_name.lower().replace('_', '_').replace('-', '_')}.json"
            output_path = output_dir / output_filename

            with open(output_path, "w") as f:
                json.dump(coverage_bins, f, indent=2)

            # Print summary
            total_files = len(file_coverage)
            print(f"  Total files: {total_files}")
            for bin_name, files in coverage_bins.items():
                print(f"  {bin_name}: {len(files)} files")
            print(f"  Saved to: {output_path}")
            print()

    def generate_summary_stats(self):
        """Generate summary statistics for all models."""
        model_configs = [
            ("Human", "mypy_results_original_files_with_errors.json"),
            ("o3_mini_1st_run", "mypy_results_o3_mini_1st_run_with_errors.json"),
            (
                "claude_3_7_sonnet",
                "mypy_results_claude3_sonnet_1st_run_with_errors.json",
            ),
            ("gpt4o", "mypy_results_gpt4o_with_errors.json"),
            ("o1-mini", "mypy_results_o1_mini_with_errors.json"),
            ("deepseek", "mypy_results_deepseek_with_errors.json"),
        ]

        summary_stats = {}

        for model_name, file_path in model_configs:
            print(f"Calculating stats for {model_name}...")

            mypy_data = self.load_mypy_results(model_name, file_path)
            if not mypy_data:
                continue

            file_coverage = self.calculate_file_coverage(mypy_data)
            coverage_values = list(file_coverage.values())

            summary_stats[model_name] = {
                "total_files": len(file_coverage),
                "mean_coverage": (
                    sum(coverage_values) / len(coverage_values)
                    if coverage_values
                    else 0
                ),
                "median_coverage": (
                    sorted(coverage_values)[len(coverage_values) // 2]
                    if coverage_values
                    else 0
                ),
                "min_coverage": min(coverage_values) if coverage_values else 0,
                "max_coverage": max(coverage_values) if coverage_values else 0,
                "files_0_20_percent": len([c for c in coverage_values if c < 20]),
                "files_20_50_percent": len(
                    [c for c in coverage_values if 20 <= c < 50]
                ),
                "files_50_80_percent": len(
                    [c for c in coverage_values if 50 <= c < 80]
                ),
                "files_80_100_percent": len([c for c in coverage_values if c >= 80]),
            }

        # Save summary stats
        output_dir = Path("type_coverage_bins")
        with open(output_dir / "coverage_summary_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=2)

        print(f"\nSummary stats saved to: {output_dir / 'coverage_summary_stats.json'}")

        # Print summary
        print("\n=== COVERAGE SUMMARY ===")
        for model, stats in summary_stats.items():
            print(f"\n{model}:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Mean coverage: {stats['mean_coverage']:.2f}%")
            print(f"  Median coverage: {stats['median_coverage']:.2f}%")
            print(f"  0-20%: {stats['files_0_20_percent']} files")
            print(f"  20-50%: {stats['files_20_50_percent']} files")
            print(f"  50-80%: {stats['files_50_80_percent']} files")
            print(f"  80-100%: {stats['files_80_100_percent']} files")


def main():
    # Initialize generator
    mypy_results_dir = "mypy_outputs"
    generator = CoverageBinsGenerator(mypy_results_dir)

    # Generate coverage bins for all models
    print("Generating coverage bins for all models...")
    generator.generate_all_models()

    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generator.generate_summary_stats()

    print("\nAll files generated in 'type_coverage_bins/' directory")


if __name__ == "__main__":
    main()
