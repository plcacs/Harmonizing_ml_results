import json


def load_results():
    """Load the analysis results."""
    with open("any_analysis_results.json", "r") as f:
        return json.load(f)


def print_key_findings(results):
    """Print key findings from the analysis."""
    human = results["human_analysis"]
    o1_mini = results["o1_mini_analysis"]
    comparison = results["comparison"]

    print("=" * 60)
    print("ANY TYPE ANALYSIS - KEY FINDINGS")
    print("=" * 60)

    # Overall comparison
    print(f"\n1. OVERALL ANY USAGE:")
    print(
        f"   Human: {human['overall_any_percentage']:.2f}% ({human['total_any_annotations']:,} out of {human['total_annotations']:,})"
    )
    print(
        f"   O1-Mini: {o1_mini['overall_any_percentage']:.2f}% ({o1_mini['total_any_annotations']:,} out of {o1_mini['total_annotations']:,})"
    )
    print(
        f"   Difference: {comparison['overall_comparison']['difference']:.2f}% (O1-Mini uses {comparison['overall_comparison']['difference']/human['overall_any_percentage']*100:.1f}% more Any types)"
    )

    # Top Any types comparison
    print(f"\n2. TOP ANY TYPES COMPARISON:")
    human_top = sorted(
        human["any_type_distribution"].items(), key=lambda x: x[1], reverse=True
    )[:5]
    o1_mini_top = sorted(
        o1_mini["any_type_distribution"].items(), key=lambda x: x[1], reverse=True
    )[:5]

    print("   Human top 5:")
    for type_name, count in human_top:
        print(f"     {type_name}: {count:,}")

    print("   O1-Mini top 5:")
    for type_name, count in o1_mini_top:
        print(f"     {type_name}: {count:,}")

    # Coverage bin analysis
    print(f"\n3. COVERAGE BIN ANALYSIS:")
    print("   Bins with highest Any usage:")

    # Sort bins by Any percentage
    bin_data = []
    for bin_name, bin_info in comparison["bin_comparison"].items():
        if bin_info["human_files"] > 0 or bin_info["o1_mini_files"] > 0:
            bin_data.append(
                {
                    "bin": bin_name,
                    "human_pct": bin_info["human_any_percentage"],
                    "o1_mini_pct": bin_info["o1_mini_any_percentage"],
                    "difference": bin_info["difference"],
                    "human_files": bin_info["human_files"],
                    "o1_mini_files": bin_info["o1_mini_files"],
                }
            )

    # Sort by O1-Mini percentage (highest first)
    bin_data.sort(key=lambda x: x["o1_mini_pct"], reverse=True)

    for bin_info in bin_data[:5]:
        print(
            f"     {bin_info['bin']}: Human {bin_info['human_pct']:.2f}% vs O1-Mini {bin_info['o1_mini_pct']:.2f}% (diff: {bin_info['difference']:.2f}%)"
        )

    # Most problematic bins
    print(f"\n4. MOST PROBLEMATIC BINS (highest difference):")
    bin_data.sort(key=lambda x: x["difference"], reverse=True)

    for bin_info in bin_data[:5]:
        print(
            f"     {bin_info['bin']}: O1-Mini uses {bin_info['difference']:.2f}% more Any types than Human"
        )

    # Files with most Any usage
    print(f"\n5. FILES WITH MOST ANY USAGE:")
    print("   (This would require additional analysis of individual files)")


def create_visualization_data(results):
    """Create data for visualization."""
    bin_data = []
    for bin_name, bin_info in results["comparison"]["bin_comparison"].items():
        if bin_info["human_files"] > 0 or bin_info["o1_mini_files"] > 0:
            bin_data.append(
                {
                    "bin": bin_name,
                    "human_pct": bin_info["human_any_percentage"],
                    "o1_mini_pct": bin_info["o1_mini_any_percentage"],
                    "difference": bin_info["difference"],
                }
            )

    return bin_data


def print_detailed_analysis(results):
    """Print detailed analysis by category."""
    human = results["human_analysis"]
    o1_mini = results["o1_mini_analysis"]

    print(f"\n6. DETAILED ANALYSIS BY CATEGORY:")

    categories = [
        "function_params",
        "function_returns",
        "variables",
        "class_attributes",
    ]

    for category in categories:
        human_cat = human["file_analyses"]
        o1_mini_cat = o1_mini["file_analyses"]

        # Aggregate category data
        human_total = sum(
            file_data[category]["total"] for file_data in human_cat.values()
        )
        human_any = sum(file_data[category]["any"] for file_data in human_cat.values())
        o1_mini_total = sum(
            file_data[category]["total"] for file_data in o1_mini_cat.values()
        )
        o1_mini_any = sum(
            file_data[category]["any"] for file_data in o1_mini_cat.values()
        )

        human_pct = (human_any / human_total * 100) if human_total > 0 else 0
        o1_mini_pct = (o1_mini_any / o1_mini_total * 100) if o1_mini_total > 0 else 0

        print(f"   {category.replace('_', ' ').title()}:")
        print(f"     Human: {human_pct:.2f}% ({human_any:,}/{human_total:,})")
        print(f"     O1-Mini: {o1_mini_pct:.2f}% ({o1_mini_any:,}/{o1_mini_total:,})")
        print(f"     Difference: {o1_mini_pct - human_pct:.2f}%")


def main():
    """Main function to run the summary analysis."""
    try:
        results = load_results()
        print_key_findings(results)
        print_detailed_analysis(results)

        # Create visualization data
        bin_data = create_visualization_data(results)
        print(f"\n7. VISUALIZATION DATA READY:")
        print(f"   Coverage bins: {len(bin_data)}")

        # Save summary to file
        with open("any_analysis_summary.txt", "w") as f:
            f.write("ANY TYPE ANALYSIS SUMMARY\n")
            f.write("=" * 40 + "\n\n")

            human = results["human_analysis"]
            o1_mini = results["o1_mini_analysis"]
            comparison = results["comparison"]

            f.write(f"Overall Any Usage:\n")
            f.write(f"Human: {human['overall_any_percentage']:.2f}%\n")
            f.write(f"O1-Mini: {o1_mini['overall_any_percentage']:.2f}%\n")
            f.write(
                f"Difference: {comparison['overall_comparison']['difference']:.2f}%\n\n"
            )

            f.write("Key Findings:\n")
            f.write(
                f"1. O1-Mini uses {comparison['overall_comparison']['difference']:.2f}% more Any types than Human\n"
            )
            f.write(
                f"2. O1-Mini has {o1_mini['total_annotations']/human['total_annotations']:.1f}x more total annotations\n"
            )
            f.write(
                f"3. O1-Mini has {o1_mini['total_any_annotations']/human['total_any_annotations']:.1f}x more Any annotations\n"
            )

        print(f"\nSummary saved to any_analysis_summary.txt")

    except FileNotFoundError:
        print(
            "Error: any_analysis_results.json not found. Please run any_type_analysis.py first."
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
