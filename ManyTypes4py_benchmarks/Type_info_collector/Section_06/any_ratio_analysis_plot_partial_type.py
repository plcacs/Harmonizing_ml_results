import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configuration for LLMs including Human baseline
LLM_CONFIGS = {
    
    "o3-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_partially_typed_files_benchmarks.json",
    },
    "deepseek": {
        "type_info_path": "../Type_info_LLMS/Type_info_deepseek_partially_typed_files_benchmarks.json",
    },
    "claude3-sonnet": {
        "type_info_path": "../Type_info_LLMS/Type_info_claude3_sonnet_partially_typed_files_benchmarks.json",
    },
    "Human": {
        "type_info_path": "../Type_info_LLMS/Type_info_original_files.json",
    }
}
# Optional filtering: consider only files with isCompiled == true from mypy results
FILTER_TO_COMPILED: bool = True  # set True to enable filtering
COMPILED_RESULTS_PATH: str = "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"

def load_json(path: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def calculate_any_ratio(type_info: Dict) -> float:
    """Calculate the ratio of 'Any' (including empty/missing) types in a file."""
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
            if isinstance(tlist, list):
                if tlist:
                    t0 = tlist[0]
                    if isinstance(t0, str) and t0.strip():
                        total_types += 1
                        if t0.strip().lower() in ["any", "tp.any", "typing.any"]:
                            any_types += 1
                    else:
                        total_types += 1
                        any_types += 1
                else:
                    total_types += 1
                    any_types += 1
    
    return (any_types / total_types * 100) if total_types > 0 else 0.0

def load_compiled_true_files(path: str) -> Tuple[Set[str], Set[str]]:
    """Load filenames where isCompiled == true. Returns (full_paths, basenames)."""
    data = load_json(path)
    full_paths: Set[str] = set()
    basenames: Set[str] = set()

    def add_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            return
        norm = name.replace("\\", "/")
        full_paths.add(norm)
        basenames.add(os.path.basename(norm))

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("isCompiled") is True:
                name = entry.get("filename") or entry.get("file") or entry.get("path")
                add_name(name)
    elif isinstance(data, dict):
        for name, entry in data.items():
            if isinstance(entry, dict) and entry.get("isCompiled") is True:
                add_name(name)

    return full_paths, basenames

def categorize_files_by_any_ratio(any_ratios: List[float]) -> Dict[str, int]:
    """Categorize files by their Any ratio."""
    categories = {
        "0%": 0,
        "1-10%": 0,
        "11-20%": 0,
        "21-30%": 0,
        "31-40%": 0,
        "41-50%": 0,
        "51-60%": 0,
        ">60%": 0
    }
    
    for ratio in any_ratios:
        if ratio == 0:
            categories["0%"] += 1
        elif 1 <= ratio <= 10:
            categories["1-10%"] += 1
        elif 11 <= ratio <= 20:
            categories["11-20%"] += 1
        elif 21 <= ratio <= 30:
            categories["21-30%"] += 1
        elif 31 <= ratio <= 40:
            categories["31-40%"] += 1
        elif 41 <= ratio <= 50:
            categories["41-50%"] += 1
        elif 51 <= ratio <= 60:
            categories["51-60%"] += 1
        else:
            categories[">60%"] += 1
    
    return categories

def main():
    """Main analysis function for Any ratio comparison."""
    print("Loading LLM data...")
    
    # Load all LLM data
    llm_data = {}
    for llm_name, config in LLM_CONFIGS.items():
        type_info = load_json(config["type_info_path"])
        llm_data[llm_name] = type_info
        print(f"Loaded {llm_name}: {len(type_info)} files")

    # Optional: filter to compiled files only
    compiled_full: Set[str] = set()
    compiled_base: Set[str] = set()
    if FILTER_TO_COMPILED:
        compiled_full, compiled_base = load_compiled_true_files(COMPILED_RESULTS_PATH)
        print(f"Filtering to compiled files: {len(compiled_full)} entries")
    
    # Calculate Any ratios for each LLM
    llm_any_ratios = {}
    for llm_name, type_info in llm_data.items():
        print(f"Calculating Any ratios for {llm_name}...")
        any_ratios = []
        
        for filename, file_info in type_info.items():
            if FILTER_TO_COMPILED:
                norm_name = filename.replace("\\", "/")
                base = os.path.basename(norm_name)
                if norm_name not in compiled_full and base not in compiled_base:
                    continue
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
        "1-10%",
        "11-20%",
        "21-30%",
        "31-40%",
        "41-50%",
        "51-60%",
        ">60%"
    ]
    
    # Ensure Human comes first in plotting/legend order
    llm_names = ["Human"] + [n for n in LLM_CONFIGS.keys() if n != "Human"]
    
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
        color = color_map.get(llm_name, "#666666")  # Default gray if not found
        ax.bar(x + i * width, values, width, label=llm_name, color=color, alpha=0.8)
    
    # Customize the plot
    #ax.set_xlabel('File Categorization by Any Ratio', fontsize=18)
    ax.set_ylabel('Percentage of Total Files (%)', fontsize=18)
    # ax.set_title('"Any Usage Distribution: LLMs Relative to Human Annotations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 3)  # Center the x-tick labels (7 LLMs, so center at 3rd position)
    ax.set_xticklabels(category_names, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 60)
    ax.legend(loc='upper center', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, category in enumerate(category_names):
        for j, llm_name in enumerate(llm_names):
            value = plot_data[i][j]
            if value > 0:  # Only show labels for non-zero values
                ax.text(i + j * width, value + 1.0, f'{value:.1f}%', 
                       ha='center', va='bottom', fontsize=14,rotation=90)
    
    plt.tight_layout()
    plt.savefig("any_usage_distribution_llms_relative_to_human_annotations_partial_user_annotated.pdf", bbox_inches="tight")
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
