import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# Configuration for LLMs including Human baseline
LLM_CONFIGS = {
    
    "o3-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
    },
    "deepseek": {
        "type_info_path": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
    },
    "claude3-sonnet": {
        "type_info_path": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
    },
    "o3-mini-renamed": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
    },
    "deepseek-renamed": {
        "type_info_path": "../Type_info_LLMS/Type_info_deepseek_renamed_output_2_benchmarks.json",
    },
    "claude3-sonnet-renamed": {
        "type_info_path": "../Type_info_LLMS/Type_info_claude_sonnet_renamed_output_benchmarks.json",
    },
    "gpt-5": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt5_1st_run_benchmarks.json",
    },
    "Human": {
        "type_info_path": "../Type_info_LLMS/Type_info_original_files.json",
    }
}

def get_base_filenames() -> set:
    """Get the set of Python filenames from Hundrad_renamed_benchmarks directory."""
    base_dir = "../../Hundrad_renamed_benchmarks"
    base_filenames = set()
    
    try:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py'):
                    base_filenames.add(file)
        print(f"Found {len(base_filenames)} base Python files")
        return base_filenames
    except Exception as e:
        print(f"Error reading base directory {base_dir}: {e}")
        return set()

def load_json(path: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def calculate_any_ratio(type_info: Dict) -> float:
    """Calculate the ratio of 'Any' types in a file."""
    total_types = 0
    any_types = 0
    
    if not isinstance(type_info, dict):
        return 0.0
    
    for func_name, items in type_info.items():
        if not isinstance(items, list):
            continue
            
        for entry in items:
            if not isinstance(entry, dict):
                continue
                
            tlist = entry.get("type", [])
            if isinstance(tlist, list) and tlist:
                t0 = tlist[0]
                if isinstance(t0, str) and t0.strip():
                    total_types += 1
                    # Check if it's an "Any" type
                    if t0.strip().lower() in ["any", "tp.any", "typing.any"]:
                        any_types += 1
    
    return (any_types / total_types * 100) if total_types > 0 else 0.0

def categorize_files_by_any_ratio(any_ratios: List[float]) -> Dict[str, int]:
    """Categorize files by their Any ratio."""
    categories = {
        "0%": 0,
        "1-5%": 0,
        "6-10%": 0,
        "11-15%": 0,
        "16-20%": 0,
        "21-30%": 0,
        ">30%": 0
    }
    
    for ratio in any_ratios:
        if ratio == 0:
            categories["0%"] += 1
        elif ratio >= 1 and ratio <= 5:
            categories["1-5%"] += 1
        elif ratio >= 6 and ratio <= 10:
            categories["6-10%"] += 1
        elif ratio >= 11 and ratio <= 15:
            categories["11-15%"] += 1
        elif ratio >= 16 and ratio <= 20:
            categories["16-20%"] += 1
        elif ratio >= 21 and ratio <= 30:
            categories["21-30%"] += 1
        else:
            categories[">30%"] += 1
    
    return categories

def main():
    """Main analysis function for Any ratio comparison."""
    print("Loading LLM data...")
    
    # Get base filenames to filter by
    base_filenames = get_base_filenames()
    if not base_filenames:
        print("No base files found. Exiting.")
        return
    
    # Load all LLM data
    llm_data = {}
    for llm_name, config in LLM_CONFIGS.items():
        type_info = load_json(config["type_info_path"])
        # Filter to only include files that exist in base directory
        filtered_type_info = {k: v for k, v in type_info.items() if k in base_filenames}
        llm_data[llm_name] = filtered_type_info
        print(f"Loaded {llm_name}: {len(filtered_type_info)} files (filtered from {len(type_info)})")
    
    # Calculate Any ratios for each LLM
    llm_any_ratios = {}
    for llm_name, type_info in llm_data.items():
        print(f"Calculating Any ratios for {llm_name}...")
        any_ratios = []
        
        for filename, file_info in type_info.items():
            ratio = calculate_any_ratio(file_info)
            any_ratios.append(ratio)
        
        llm_any_ratios[llm_name] = any_ratios
        print(f"  {llm_name}: {len(any_ratios)} files processed")
    
    # Categorize files for each LLM
    llm_categories = {}
    for llm_name, ratios in llm_any_ratios.items():
        categories = categorize_files_by_any_ratio(ratios)
        llm_categories[llm_name] = categories
    
    # Prepare data for plotting
    category_names = [
        "0%",
        "1-5%",
        "6-10%",
        "11-15%",
        "16-20%",
        "21-30%",
        ">30%"
    ]
    
    llm_names = list(LLM_CONFIGS.keys())
    
    # Calculate percentages for each LLM and category
    plot_data = []
    for category in category_names:
        category_data = []
        for llm_name in llm_names:
            total_files = len(llm_any_ratios[llm_name])
            if total_files > 0:
                percentage = (llm_categories[llm_name][category] / total_files) * 100
            else:
                percentage = 0
            category_data.append(percentage)
        plot_data.append(category_data)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(category_names))
    width = 0.11  # Width of bars (adjusted for 7 categories)
    
    # Standardized color scheme for all bar plots
    color_map = {
        "deepseek": "blue",      # Blue
        "gpt-4o": "orange",        # Orange  
        "o1-mini": "skyblue",       # Sky Blue
        "gpt-3.5": "green",       # Green
        "Human": "pink",          # Yellow
        "o3-mini": "red",        # Dark Red
        "claude3-sonnet": "purple"  # Purple
    }
    
    # Create bars for each LLM
    for i, llm_name in enumerate(llm_names):
        values = [plot_data[j][i] for j in range(len(category_names))]
        base_color = color_map.get(llm_name.replace("-renamed", ""), "#666666")
        
        # Add pattern for renamed versions
        if "-renamed" in llm_name:
            ax.bar(x + i * width, values, width, label=llm_name, color=base_color, alpha=0.8, 
                   hatch='///', edgecolor='black', linewidth=0.5)
        else:
            ax.bar(x + i * width, values, width, label=llm_name, color=base_color, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('File Categorization by Any Ratio', fontsize=16)
    ax.set_ylabel('Percentage of Total Files (%)', fontsize=16)
    # ax.set_title('"Any Usage Distribution: LLMs Relative to Human Annotations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 3)  # Center the x-tick labels (7 LLMs, so center at 3rd position)
    ax.set_xticklabels(category_names, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend( loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, category in enumerate(category_names):
        for j, llm_name in enumerate(llm_names):
            value = plot_data[i][j]
            if value > 0:  # Only show labels for non-zero values
                ax.text(i + j * width, value + 1.0, f'{value:.1f}%', 
                       ha='center', va='bottom', fontsize=12,rotation=90)
    
    plt.tight_layout()
    plt.savefig("any_usage_distribution_llms_relative_to_human_annotations.pdf", bbox_inches="tight")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for llm_name in llm_names:
        print(f"\n{llm_name}:")
        total_files = len(llm_any_ratios[llm_name])
        avg_any_ratio = np.mean(llm_any_ratios[llm_name])
        print(f"  Total files: {total_files}")
        print(f"  Average Any ratio: {avg_any_ratio:.2f}%")
        
        for category in category_names:
            count = llm_categories[llm_name][category]
            percentage = (count / total_files * 100) if total_files > 0 else 0
            print(f"  {category}: {count} files ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
