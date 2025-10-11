import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean
from typing import Dict, List

# Input and output files
DETAILED_CSV = "./precision_results/llm_binary_precision_detailed.csv"
#PLOT_PDF = "./precision_results/llm_binary_precision_plot.pdf"
#PLOT_PERCENT_PDF = "./precision_results/llm_precision_plot_percent.pdf"
#PLOT_NORMALIZED_PDF = "./precision_results/llm_precision_plot_normalized.pdf"
PLOT_MERGED_PDF = "./precision_results/llm_binary_precision_plot_merged_percent_and_normalized.pdf"


def read_detailed_csv(path: str) -> List[Dict[str, object]]:
    """Read the detailed precision CSV and extract data grouped by winner group size."""
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            winners_raw = row["winners"].strip()
            winners = [w for w in winners_raw.split(";") if w]
            group_size = max(1, len(winners))

            # Parse scores
            scores_str = row["scores..."]
            scores = {}
            for score_pair in scores_str.split(";"):
                if ":" in score_pair:
                    llm, score_str = score_pair.split(":", 1)
                    try:
                        scores[llm] = float(score_str)
                    except ValueError:
                        continue

            rows.append(
                {
                    "filename": filename,
                    "winners": winners,
                    "group_size": group_size,
                    "scores": scores,
                }
            )

    return rows


def build_file_counts_by_group(
    rows: List[Dict[str, object]],
) -> Dict[int, Dict[str, int]]:
    """Group files by winner group size and count files per LLM within each group."""
    grouped_counts = defaultdict(lambda: defaultdict(int))

    for row in rows:
        group_size = int(row["group_size"])
        winners = row["winners"]  # type: ignore

        # Count each winner LLM for this group size
        for llm in winners:
            grouped_counts[group_size][llm] += 1

    result = {}
    for group_size, llm_dict in grouped_counts.items():
        result[group_size] = dict(llm_dict)

    return result


def build_percentage_by_group(
    rows: List[Dict[str, object]],
) -> Dict[int, Dict[str, float]]:
    """Group files by winner group size and compute percentage of files per LLM within each group."""
    grouped_counts = defaultdict(lambda: defaultdict(int))

    for row in rows:
        group_size = int(row["group_size"])
        winners = row["winners"]  # type: ignore

        # Count each winner LLM for this group size
        for llm in winners:
            grouped_counts[group_size][llm] += 1

    result = {}
    for group_size, llm_dict in grouped_counts.items():
        total_files = sum(llm_dict.values())
        if total_files > 0:
            result[group_size] = {
                llm: (count / total_files) * 100 for llm, count in llm_dict.items()
            }
        else:
            result[group_size] = {}

    return result


def build_normalized_by_group(
    rows: List[Dict[str, object]],
) -> Dict[int, Dict[str, float]]:
    """Group files by winner group size and normalize wins by total LLM success count."""
    grouped_counts = defaultdict(lambda: defaultdict(int))
    total_llm_wins = defaultdict(int)

    # First pass: count total wins per LLM
    for row in rows:
        winners = row["winners"]  # type: ignore
        for llm in winners:
            total_llm_wins[llm] += 1

    # Second pass: count wins per group size
    for row in rows:
        group_size = int(row["group_size"])
        winners = row["winners"]  # type: ignore
        for llm in winners:
            grouped_counts[group_size][llm] += 1

    result = {}
    for group_size, llm_dict in grouped_counts.items():
        result[group_size] = {}
        for llm, count in llm_dict.items():
            total_wins = total_llm_wins.get(llm, 0)
            if total_wins > 0:
                result[group_size][llm] = (count / total_wins) * 100
            else:
                result[group_size][llm] = 0.0

    return result


# Custom color map for consistent coloring
color_map = {
    "gpt-3.5": "green",
    "gpt-4o": "orange", 
    "o1-mini": "skyblue",
    "o3-mini": "red",
    "claude3-sonnet": "purple",
    "deepseek": "blue"
}

def create_hatch_pattern(color, is_normalized=False):
    """Create color and hatch pattern for bars."""
    if is_normalized:
        return color, '///'  # Vertical stripes for normalized
    else:
        return color, None   # Solid for percent

def get_legend_name(llm_name):
    """Convert internal LLM names to proper legend names."""
    legend_names = {
        "gpt-3.5": "gpt-3.5",
        "gpt-4o": "gpt-4o",
        "o1-mini": "o1-mini",
        "o3-mini": "o3-mini",
        "claude3-sonnet": "claude3 sonnet",
        "deepseek": "deepseek"
    }
    return legend_names.get(llm_name, llm_name)

def merged_grouped_bar(
    percent_data: Dict[int, Dict[str, float]], 
    normalized_data: Dict[int, Dict[str, float]], 
    outfile: str
):
    """Create a merged grouped bar chart with both percent and normalized data."""
    groups = sorted(percent_data.keys())

    # Maintain specific LLM order
    llm_order = [
        "gpt-3.5",
        "gpt-4o", 
        "o1-mini",
        "o3-mini",
        "deepseek",
        "claude3-sonnet",
    ]
    llms = [llm for llm in llm_order if any(llm in percent_data[g] for g in groups)]

    x = range(len(groups))
    num_llms = len(llms)
    total_width = 0.82
    bar_width = total_width / max(1, num_llms * 2)  # Half width for 12 bars total (6 models * 2)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(num_llms * 2)]

    plt.figure(figsize=(16, 8))
    bars_by_series = []

    # Plot percent bars (solid)
    for idx, llm in enumerate(llms):
        heights = [percent_data[g].get(llm, 0.0) for g in groups]
        positions = [i + offsets[idx] for i in x]
        color, hatch = create_hatch_pattern(color_map.get(llm, "gray"), False)
        
        # Format legend name
        legend_name = get_legend_name(llm)
        bars = plt.bar(
            positions, heights, width=bar_width, 
            label=f"{legend_name} (percent)", color=color, hatch=hatch
        )
        bars_by_series.append(bars)

    # Plot normalized bars (striped)
    for idx, llm in enumerate(llms):
        heights = [normalized_data[g].get(llm, 0.0) for g in groups]
        positions = [i + offsets[idx + num_llms] for i in x]
        color, hatch = create_hatch_pattern(color_map.get(llm, "gray"), True)
        
        # Format legend name
        legend_name = get_legend_name(llm)
        bars = plt.bar(
            positions, heights, width=bar_width,
            label=f"{legend_name} (normalized)", color=color, hatch=hatch
        )
        bars_by_series.append(bars)

    # Add value labels
    total_bars = len(groups) * num_llms * 2
    if total_bars <= 200:
        for bars in bars_by_series:
            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    plt.annotate(
                        f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        rotation=90,
                    )

    plt.xticks(list(x), [str(g) for g in groups], rotation=30, ha="right", fontsize=14)
    #plt.xlabel("Winner group size (number of top scorers)", fontsize=18)
    plt.ylabel("Percentage (%)", fontsize=18)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()

def grouped_bar(
    data: Dict[int, Dict[str, float]], ylabel: str, title: str, outfile: str
):
    """Create a grouped bar chart similar to the existing ones."""
    groups = sorted(data.keys())

    # Maintain specific LLM order
    llm_order = [
        "gpt-3.5",
        "gpt-4o",
        "o1-mini",
        "o3-mini",
        "deepseek",
        "claude3-sonnet",
    ]
    llms = [llm for llm in llm_order if any(llm in data[g] for g in groups)]

    x = range(len(groups))
    num_llms = len(llms)
    total_width = 0.82
    bar_width = total_width / max(1, num_llms)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(num_llms)]

    plt.figure(figsize=(14, 7))
    color_cycle = plt.get_cmap("tab10")
    bars_by_series = []
    all_heights = []

    for idx, llm in enumerate(llms):
        heights = [data[g].get(llm, 0.0) for g in groups]
        all_heights.extend(heights)
        positions = [i + offsets[idx] for i in x]
        
        # Format legend name
        legend_name = get_legend_name(llm)
        bars = plt.bar(
            positions, heights, width=bar_width, label=legend_name, color=color_cycle(idx % 10)
        )
        bars_by_series.append(bars)

    # Add value labels
    total_bars = len(groups) * num_llms
    if total_bars <= 150:
        for bars in bars_by_series:
            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    plt.annotate(
                        f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        rotation=90,
                    )

    plt.xticks(list(x), [str(g) for g in groups], rotation=30, ha="right")
    #plt.xlabel("Winner group size (number of top scorers)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


def main():
    """Main function to create the precision plot."""
    print("Reading detailed precision data...")

    try:
        rows = read_detailed_csv(DETAILED_CSV)

        if not rows:
            print("No data found in the CSV file.")
            return

        print(f"Found data for {len(rows)} files")

        # Build percentage and normalized data by group
        percentage_by_group = build_percentage_by_group(rows)
        normalized_by_group = build_normalized_by_group(rows)

        # Create merged plot only
        merged_grouped_bar(
            percentage_by_group,
            normalized_by_group,
            PLOT_MERGED_PDF
        )
        print(f"Merged plot saved to: {PLOT_MERGED_PDF}")

        # Print detailed summary
        print("\nDetailed breakdown by group size:")
        print("=" * 80)
        
        for group_size in sorted(percentage_by_group.keys()):
            total_files = len([r for r in rows if r["group_size"] == group_size])
            print(f"\nGroup size {group_size} (Total files: {total_files}):")
            print("-" * 50)
            
            # Get detailed counts for this group size
            group_data = percentage_by_group[group_size]
            for llm in sorted(group_data.keys()):
                percentage = group_data[llm]
                # Calculate actual file count from percentage
                file_count = int((percentage / 100) * total_files)
                print(f"  {llm:<15}: {file_count:>3} files ({percentage:>5.1f}%)")

        print("\n" + "=" * 80)
        print("Total wins per LLM across all group sizes:")
        print("-" * 50)
        total_wins = defaultdict(int)
        for row in rows:
            for winner in row["winners"]:
                total_wins[winner] += 1

        for llm in sorted(total_wins.keys()):
            print(f"  {llm:<15}: {total_wins[llm]:>3} total wins")

    except FileNotFoundError:
        print(f"Error: Could not find {DETAILED_CSV}")
        print("Please run the precision comparison script first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
