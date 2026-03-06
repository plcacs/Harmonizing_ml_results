import json
import os
import matplotlib.pyplot as plt
import numpy as np

PERCENTAGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PERCENT_NAMES = [
    "zero", "ten", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "fully",
]

LLM_CONFIGS = {
    "deepseek": {
        "base_dir": "deepseek_outputs",
        "prefix": "mypy_results_deepseek_",
        "suffix": "_percent_typed_output.json",
    },
    "o3_mini": {
        "base_dir": "o3_mini_outputs",
        "prefix": "mypy_results_o3_mini_",
        "suffix": "_percent_typed_output.json",
    },
}

NON_TYPE_ERROR_CODES = [
    "name-defined", "import", "syntax", "no-redef", "unused-ignore",
    "override-without-super", "redundant-cast", "literal-required",
    "typeddict-unknown-key", "typeddict-item", "truthy-function",
    "str-bytes-safe", "unused-coroutine", "explicit-override",
    "truthy-iterable", "redundant-self", "redundant-await", "unreachable",
]


def has_non_type_error(errors):
    for error in errors:
        if any(t in error.lower() for t in ["syntax", "empty_body", "name_defined"]):
            return True
        if "[" in error and "]" in error:
            code = error[error.rindex("[") + 1 : error.rindex("]")]
            if code in NON_TYPE_ERROR_CODES:
                return True
    return False


def load_all_data(base_dir, prefix, suffix):
    all_data = {}
    for pct, name in zip(PERCENTAGES, PERCENT_NAMES):
        path = os.path.join(base_dir, f"{prefix}{name}{suffix}")
        with open(path, "r") as f:
            all_data[pct] = json.load(f)
    return all_data


def build_per_file_table(all_data):
    """Build {filename: {pct: error_count}} after filtering non-type errors."""
    all_files = set()
    for pct in PERCENTAGES:
        all_files.update(all_data[pct].keys())

    table = {}
    for fname in sorted(all_files):
        skip = False
        for pct in PERCENTAGES:
            entry = all_data[pct].get(fname)
            if entry and has_non_type_error(entry.get("errors", [])):
                skip = True
                break
        if skip:
            continue
        table[fname] = {pct: all_data[pct][fname]["error_count"]
                        for pct in PERCENTAGES if fname in all_data[pct]}
    return table


def min_percent_to_compile(all_data, table):
    """For each file, find the lowest % where error_count == 0. None if never."""
    result = {}
    for fname, errors_by_pct in table.items():
        result[fname] = None
        for pct in PERCENTAGES:
            if errors_by_pct.get(pct, -1) == 0:
                result[fname] = pct
                break
    return result


def categorize_files(table, min_pct):
    categories = {
        "always_pass": [],    # 0 errors at all %
        "easy": [],           # first passes at <= 30%
        "medium": [],         # first passes at 40-70%
        "hard": [],           # first passes at 80-100%
        "stubborn": [],       # never passes
        "non_monotonic": [],  # errors increase at some step
    }
    for fname, errors_by_pct in table.items():
        pcts_present = sorted(errors_by_pct.keys())
        errs = [errors_by_pct[p] for p in pcts_present]

        if all(e == 0 for e in errs):
            categories["always_pass"].append(fname)
        elif min_pct[fname] is None:
            categories["stubborn"].append(fname)
        elif min_pct[fname] <= 30:
            categories["easy"].append(fname)
        elif min_pct[fname] <= 70:
            categories["medium"].append(fname)
        else:
            categories["hard"].append(fname)

        for i in range(1, len(errs)):
            if errs[i] > errs[i - 1]:
                categories["non_monotonic"].append(fname)
                break

    return categories


# --- Plot 1: Histogram of minimum % to compile ---
def plot_histogram(all_min_pcts):
    fig, axes = plt.subplots(1, len(all_min_pcts), figsize=(7 * len(all_min_pcts), 5),
                             squeeze=False)
    for idx, (llm_name, min_pct) in enumerate(all_min_pcts.items()):
        values = [v for v in min_pct.values() if v is not None]
        never_count = sum(1 for v in min_pct.values() if v is None)
        ax = axes[0][idx]
        ax.hist(values, bins=PERCENTAGES + [110], edgecolor="black",
                align="left", rwidth=0.8)
        ax.set_xticks(PERCENTAGES)
        ax.set_xlabel("Minimum Annotation % to Pass Mypy")
        ax.set_ylabel("Number of Files")
        ax.set_title(f"{llm_name}\n(never pass: {never_count} files)")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_min_percent_histogram.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


# --- Plot 3: Box plot of error counts per % ---
def plot_boxplot(all_tables):
    fig, axes = plt.subplots(1, len(all_tables), figsize=(7 * len(all_tables), 5),
                             squeeze=False)
    for idx, (llm_name, table) in enumerate(all_tables.items()):
        data_by_pct = []
        for pct in PERCENTAGES:
            data_by_pct.append([table[f].get(pct, 0) for f in table])
        ax = axes[0][idx]
        bp = ax.boxplot(data_by_pct, labels=[str(p) for p in PERCENTAGES],
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("skyblue")
        ax.set_xlabel("Annotation %")
        ax.set_ylabel("Mypy Error Count")
        ax.set_title(f"{llm_name} — Error Distribution per %")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_error_boxplot.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


# --- Plot 4: Heatmap ---
def plot_heatmap(all_tables):
    for llm_name, table in all_tables.items():
        filenames = sorted(table.keys())
        matrix = np.array([[table[f].get(pct, 0) for pct in PERCENTAGES]
                           for f in filenames])
        fig, ax = plt.subplots(figsize=(10, max(6, len(filenames) * 0.08)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(len(PERCENTAGES)))
        ax.set_xticklabels([f"{p}%" for p in PERCENTAGES])
        ax.set_xlabel("Annotation %")
        ax.set_ylabel(f"Files ({len(filenames)} total)")
        ax.set_yticks([])
        ax.set_title(f"{llm_name} — Per-File Error Heatmap")
        plt.colorbar(im, ax=ax, label="Error Count")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"plot_heatmap_{llm_name}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.show()


# --- Analysis 6: Categorization table ---
def print_categories(llm_name, categories, total):
    print(f"\n{'=' * 55}")
    print(f"  {llm_name.upper()} — File Categorization ({total} files)")
    print(f"{'=' * 55}")
    print(f"  {'Category':<20} {'Count':>6} {'%':>8}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 8}")
    for cat, files in categories.items():
        pct = len(files) / total * 100 if total else 0
        print(f"  {cat:<20} {len(files):>6} {pct:>7.1f}%")


OUTPUT_DIR = "per_file_analysis_results"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_tables = {}
    all_min_pcts = {}
    all_categories = {}

    for llm_name, cfg in LLM_CONFIGS.items():
        all_data = load_all_data(cfg["base_dir"], cfg["prefix"], cfg["suffix"])
        table = build_per_file_table(all_data)
        min_pct = min_percent_to_compile(all_data, table)
        categories = categorize_files(table, min_pct)

        all_tables[llm_name] = table
        all_min_pcts[llm_name] = min_pct
        all_categories[llm_name] = categories

        print_categories(llm_name, categories, len(table))

    plot_histogram(all_min_pcts)
    plot_boxplot(all_tables)
    plot_heatmap(all_tables)
