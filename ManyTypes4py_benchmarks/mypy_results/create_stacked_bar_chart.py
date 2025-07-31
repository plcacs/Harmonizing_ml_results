#!/usr/bin/env python3
"""
Stacked Bar Chart Generator for Multiple LLMs

This script creates stacked bar charts showing the distribution of
success/failure categories (both_success, llm_only_failures, both_errors)
across coverage bins for multiple LLM models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class MultiLLMStackedBarChartGenerator:
    def __init__(self):
        # Define the models and their corresponding files
        self.models = {
            "Human": {
                "analysis_file": "analysis_outputs/analysis_original_files_simplified.json",
                "coverage_bins_file": "type_coverage_bins/coverage_bin_files_human.json",
            },
            "GPT4o": {
                "analysis_file": "analysis_outputs/analysis_gpt4o_simplified.json",
                "coverage_bins_file": "type_coverage_bins/coverage_bin_files_gpt4o.json",
            },
            "DeepSeek": {
                "analysis_file": "analysis_outputs/analysis_deepseek_simplified.json",
                "coverage_bins_file": "type_coverage_bins/coverage_bin_files_deepseek.json",
            },
            "Claude": {
                "analysis_file": "analysis_outputs/analysis_claude_3_5_sonnet_simplified.json",
                "coverage_bins_file": "type_coverage_bins/coverage_bin_files_claude_3_7_sonnet.json",
            },
            "O1-mini": {
                "analysis_file": "analysis_outputs/analysis_o1-mini_simplified.json",
                "coverage_bins_file": "type_coverage_bins/coverage_bin_files_o1_mini.json",
            },
            "O3-mini": {
                "analysis_file": "analysis_outputs/analysis_o3_mini_1st_run_simplified.json",
                "coverage_bins_file": "type_coverage_bins/coverage_bin_files_o3_mini_1st_run.json",
            },
        }

    def load_data_for_model(self, model_name: str) -> Tuple[Dict, Dict]:
        """Load analysis and coverage bins data for a specific model."""
        model_config = self.models[model_name]

        try:
            with open(model_config["analysis_file"], "r") as f:
                analysis_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Analysis file not found for {model_name}")
            return {}, {}

        try:
            with open(model_config["coverage_bins_file"], "r") as f:
                coverage_bins = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Coverage bins file not found for {model_name}")
            return analysis_data, {}

        return analysis_data, coverage_bins

    def categorize_files_by_coverage(
        self, analysis_data: Dict, coverage_bins: Dict
    ) -> Dict[str, Dict[str, int]]:
        """Categorize files by their success/failure status and coverage bin."""
        # Get file lists from analysis data
        both_success_files = set(analysis_data["files"]["both_success"])
        llm_only_failures_files = set(analysis_data["files"]["llm_only_failures"])
        both_errors_files = set(analysis_data["files"]["both_errors"])

        # Initialize results
        bin_categories = defaultdict(
            lambda: {
                "both_success": 0,
                "llm_only_failures": 0,
                "both_errors": 0,
            }
        )

        # Process each coverage bin
        for bin_name, files in coverage_bins.items():
            for file in files:
                if file in both_success_files:
                    bin_categories[bin_name]["both_success"] += 1
                elif file in llm_only_failures_files:
                    bin_categories[bin_name]["llm_only_failures"] += 1
                elif file in both_errors_files:
                    bin_categories[bin_name]["both_errors"] += 1

        return dict(bin_categories)

    def create_stacked_bar_chart(
        self, model_name: str, bin_categories: Dict[str, Dict[str, int]]
    ):
        """Create stacked bar chart with actual numbers for a specific model."""
        # Define coverage bins in order
        bin_order = [
            "0-5%",
            "5-10%",
            "10-20%",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%",
            "100%",
        ]

        # Filter to only include bins that exist in our data
        existing_bins = [
            bin_name for bin_name in bin_order if bin_name in bin_categories
        ]

        # Prepare data for plotting
        categories = [
            "both_success",
            "llm_only_failures",
            "both_errors",
        ]
        colors = [
            "#2E8B57",  # Green for success
            "#FF6B6B",  # Red for LLM failures
            "#FFD93D",  # Yellow for both errors
        ]

        # Create data arrays
        data = []
        for category in categories:
            category_data = [
                bin_categories[bin_name][category] for bin_name in existing_bins
            ]
            data.append(category_data)

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create stacked bars
        bottom = np.zeros(len(existing_bins))
        bars = []

        for i, (category, color) in enumerate(zip(categories, colors)):
            bar = ax.bar(
                existing_bins,
                data[i],
                bottom=bottom,
                label=category.replace("_", " ").title(),
                color=color,
                alpha=0.8,
            )
            bars.append(bar)
            bottom += data[i]

        # Customize the plot
        ax.set_xlabel("Coverage Bins", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Files", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Distribution of Success/Failure Categories Across Coverage Bins\n({model_name} Model) - Absolute Numbers",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        # Add value labels on bars
        for bar in bars:
            for rect in bar:
                height = rect.get_height()
                if height > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        rect.get_y() + height / 2.0,
                        f"{int(height)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        # plt.show()

        # Save the plot
        plt.savefig(
            f"type_coverage_bins/numbers_barchart/stacked_bar_chart_{model_name}.pdf",
            bbox_inches="tight",
        )

    def create_percentage_stacked_bar_chart(
        self, model_name: str, bin_categories: Dict[str, Dict[str, int]]
    ):
        """Create stacked bar chart with percentages for a specific model."""
        # Define coverage bins in order
        bin_order = [
            "0-5%",
            "5-10%",
            "10-20%",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%",
            "100%",
        ]

        # Filter to only include bins that exist in our data
        existing_bins = [
            bin_name for bin_name in bin_order if bin_name in bin_categories
        ]

        # Prepare data for plotting
        categories = [
            "both_success",
            "llm_only_failures",
            "both_errors",
        ]
        colors = [
            "#2E8B57",  # Green for success
            "#FF6B6B",  # Red for LLM failures
            "#FFD93D",  # Yellow for both errors
        ]

        # Create percentage data arrays
        percentage_data = []
        for category in categories:
            category_percentages = []
            for bin_name in existing_bins:
                total_in_bin = sum(bin_categories[bin_name].values())
                if total_in_bin > 0:
                    percentage = (
                        bin_categories[bin_name][category] / total_in_bin
                    ) * 100
                    category_percentages.append(percentage)
                else:
                    category_percentages.append(0)
            percentage_data.append(category_percentages)

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create stacked bars
        bottom = np.zeros(len(existing_bins))
        bars = []

        for i, (category, color) in enumerate(zip(categories, colors)):
            bar = ax.bar(
                existing_bins,
                percentage_data[i],
                bottom=bottom,
                label=category.replace("_", " ").title(),
                color=color,
                alpha=0.8,
            )
            bars.append(bar)
            bottom += percentage_data[i]

        # Customize the plot
        ax.set_xlabel("Coverage Bins", fontsize=12, fontweight="bold")
        ax.set_ylabel("Percentage of Files (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Distribution of Success/Failure Categories Across Coverage Bins\n({model_name} Model) - Percentages",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Set y-axis to show percentages
        ax.set_ylim(0, 100)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        # Add percentage labels on bars
        for bar in bars:
            for rect in bar:
                height = rect.get_height()
                if height > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        rect.get_y() + height / 2.0,
                        f"{height:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        # plt.show()

        # Save the plot
        plt.savefig(
            f"type_coverage_bins/percentages_barchart/stacked_bar_chart_{model_name}.pdf",
            bbox_inches="tight",
        )

    def create_comparative_chart(
        self, all_model_data: Dict[str, Dict[str, Dict[str, int]]]
    ):
        """Create a comparative chart showing success rates across models."""
        # Define coverage bins in order
        bin_order = [
            "0-5%",
            "5-10%",
            "10-20%",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%",
            "100%",
        ]

        # Get common bins across all models
        all_bins = set()
        for model_data in all_model_data.values():
            all_bins.update(model_data.keys())

        existing_bins = [bin_name for bin_name in bin_order if bin_name in all_bins]

        # Calculate success rates for each model and bin
        success_rates = {}
        for model_name, bin_categories in all_model_data.items():
            success_rates[model_name] = []
            for bin_name in existing_bins:
                if bin_name in bin_categories:
                    total_in_bin = sum(bin_categories[bin_name].values())
                    if total_in_bin > 0:
                        success_rate = (
                            bin_categories[bin_name]["both_success"] / total_in_bin
                        ) * 100
                        success_rates[model_name].append(success_rate)
                    else:
                        success_rates[model_name].append(0)
                else:
                    success_rates[model_name].append(0)

        # Create the comparative plot
        fig, ax = plt.subplots(figsize=(16, 8))

        # Define colors for different models
        model_colors = {
            "Human": "#1f77b4",
            "GPT4o": "#ff7f0e",
            "DeepSeek": "#2ca02c",
            "Claude": "#d62728",
            "O1-mini": "#9467bd",
            "O3-mini": "#8c564b",
        }

        # Plot each model
        x = np.arange(len(existing_bins))
        width = 0.15  # Width of bars
        multiplier = 0

        for model_name, success_rate_data in success_rates.items():
            if model_name in model_colors:
                offset = width * multiplier
                rects = ax.bar(
                    x + offset,
                    success_rate_data,
                    width,
                    label=model_name,
                    color=model_colors[model_name],
                    alpha=0.8,
                )
                multiplier += 1

        # Customize the plot
        ax.set_xlabel("Coverage Bins", fontsize=12, fontweight="bold")
        ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Success Rate Comparison Across Models and Coverage Bins",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(existing_bins, rotation=45, ha="right")
        ax.legend(loc="upper left", fontsize=10)
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for rects in ax.containers:
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        rect.get_y() + height / 2.0,
                        f"{height:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        plt.tight_layout()
        # plt.show()

        # Save the plot
        plt.savefig(
            f"type_coverage_bins/comparative_barchart/comparative_chart.pdf",
            bbox_inches="tight",
        )

    def generate_summary_statistics(
        self, model_name: str, bin_categories: Dict[str, Dict[str, int]]
    ):
        """Generate summary statistics for a specific model."""
        print(f"\n=== SUMMARY STATISTICS FOR {model_name.upper()} ===")

        total_files = sum(
            sum(categories.values()) for categories in bin_categories.values()
        )
        print(f"Total files analyzed: {total_files}")

        # Calculate totals for each category
        category_totals = defaultdict(int)
        for bin_name, categories in bin_categories.items():
            for category, count in categories.items():
                category_totals[category] += count

        print("\nCategory Totals:")
        for category, count in category_totals.items():
            percentage = (count / total_files) * 100
            print(
                f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
            )

        # Find bins with highest success rates
        print("\nCoverage Bins Analysis:")
        for bin_name in sorted(
            bin_categories.keys(),
            key=lambda x: (
                float(x.split("-")[0]) if "-" in x else float(x.replace("%", ""))
            ),
        ):
            categories = bin_categories[bin_name]
            total_in_bin = sum(categories.values())
            if total_in_bin > 0:
                success_rate = (categories["both_success"] / total_in_bin) * 100
                print(
                    f"  {bin_name}: {total_in_bin} files, {success_rate:.1f}% success rate"
                )


def main():
    generator = MultiLLMStackedBarChartGenerator()

    # Dictionary to store all model data for comparative analysis
    all_model_data = {}

    # Process each model
    for model_name in generator.models.keys():
        print(f"\n{'='*50}")
        print(f"Processing {model_name}...")
        print(f"{'='*50}")

        # Load data for this model
        analysis_data, coverage_bins = generator.load_data_for_model(model_name)

        if not analysis_data or not coverage_bins:
            print(f"Skipping {model_name} due to missing data files")
            continue

        # Categorize files
        bin_categories = generator.categorize_files_by_coverage(
            analysis_data, coverage_bins
        )

        # Store data for comparative analysis
        all_model_data[model_name] = bin_categories

        # Generate summary statistics
        generator.generate_summary_statistics(model_name, bin_categories)

        # Create stacked bar chart with numbers
        print(f"\nCreating stacked bar chart (numbers) for {model_name}...")
        generator.create_stacked_bar_chart(model_name, bin_categories)

        # Create stacked bar chart with percentages
        print(f"\nCreating stacked bar chart (percentages) for {model_name}...")
        generator.create_percentage_stacked_bar_chart(model_name, bin_categories)

    # Create comparative chart if we have data for multiple models
    if len(all_model_data) > 1:
        print(f"\n{'='*50}")
        print("Creating comparative chart across all models...")
        print(f"{'='*50}")
        generator.create_comparative_chart(all_model_data)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
