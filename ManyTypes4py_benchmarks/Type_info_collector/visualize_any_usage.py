import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    """Load the CSV data."""
    return pd.read_csv("coverage_bin_comparison_only_any.csv")


def create_overall_bar_chart(df):
    """Create bar chart showing overall Any usage comparison."""
    # Calculate overall averages from the data
    llms = ["Human", "O1_Mini", "DeepSeek", "GPT4o", "O3_Mini", "Claude3_Sonnet"]
    overall_percentages = []

    for llm in llms:
        pct_col = f"{llm}_Any_Pct"
        # Calculate weighted average based on file counts
        weighted_avg = np.average(df[pct_col], weights=df[f"{llm}_Files"])
        overall_percentages.append(weighted_avg)

    # Create the bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        llms,
        overall_percentages,
        color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8B4513", "#228B22"],
    )

    # Customize the chart
    plt.title(
        "Overall Any Usage Comparison Across LLMs", fontsize=16, fontweight="bold"
    )
    plt.ylabel("Any Usage Percentage (%)", fontsize=12)
    plt.xlabel("LLM Models", fontsize=12)

    # Add value labels on bars
    for bar, value in zip(bars, overall_percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    print("Bar Chart: Overall Any Usage Comparison")
    print("=" * 50)
    for llm, pct in zip(llms, overall_percentages):
        print(f"{llm}: {pct:.2f}%")

    # plt.show()
    plt.savefig(
        "visualizations/overall_any_usage_comparison_only_any.pdf", bbox_inches="tight"
    )


def create_coverage_line_chart(df):
    """Create line chart showing Any usage by coverage level."""
    # Prepare data for line chart
    llms = ["Human", "O1_Mini", "DeepSeek", "GPT4o", "O3_Mini", "Claude3_Sonnet"]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8B4513", "#228B22"]

    # Convert bin names to numeric values for x-axis
    bin_midpoints = []
    for bin_name in df["Bin"]:
        if bin_name == "100%":
            bin_midpoints.append(100)
        else:
            # Extract numbers from "0-5%" -> midpoint is 2.5
            start, end = bin_name.replace("%", "").split("-")
            midpoint = (int(start) + int(end)) / 2
            bin_midpoints.append(midpoint)

    plt.figure(figsize=(14, 8))

    # Plot line for each LLM
    for i, llm in enumerate(llms):
        pct_col = f"{llm}_Any_Pct"
        plt.plot(
            bin_midpoints,
            df[pct_col],
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors[i],
            label=llm,
        )

    # Customize the chart
    plt.title("Any Usage by Coverage Level", fontsize=16, fontweight="bold")
    plt.xlabel("Coverage Level (%)", fontsize=12)
    plt.ylabel("Any Usage Percentage (%)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set x-axis ticks to match bin midpoints
    plt.xticks(bin_midpoints, df["Bin"], rotation=45, ha="right")

    # Add grid
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print("\nLine Chart: Any Usage by Coverage Level")
    print("=" * 50)
    print("Key observations:")
    print("- Human shows consistent low usage across all levels")
    print("- O3-Mini shows highest usage across all levels")
    print("- GPT4o follows Human pattern most closely")
    print("- O1-Mini shows extreme spikes in certain bins")

    # plt.show()
    plt.savefig(
        "visualizations/any_usage_by_coverage_level_only_any.pdf", bbox_inches="tight"
    )


def main():
    """Main function to create both visualizations."""
    try:
        # Load data
        df = load_data()
        print("Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Create visualizations
        create_overall_bar_chart(df)
        create_coverage_line_chart(df)

    except FileNotFoundError:
        print("Error: coverage_bin_comparison.csv not found!")
        print("Please run any_type_analysis.py first to generate the CSV file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
