import json
import os
import shutil
from typing import Dict, List, Optional


def load_json_list_map(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_file_recursive(root_dir: str, filename: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def copy_from_run_recursive(run_dir: str, filename: str, dest_dir: str) -> bool:
    src_path = find_file_recursive(run_dir, filename)
    if src_path and os.path.isfile(src_path):
        ensure_dir(dest_dir)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(src_path, dest_path)
        return True
    return False


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))

    json_input = os.path.join(
        repo_root,
        "mypy_results",
        "Section_04",
        "both_success_files.json",
    )

    data = load_json_list_map(json_input)

    # Simple fixed mapping from run key prefixes to llm bucket and run directories
    run_to_llm = {
        # GPT-3.5
        "gpt35_1st_run": ("gpt35", os.path.join(repo_root, "gpt35_1st_run")),
        "gpt35_2nd_run": ("gpt35", os.path.join(repo_root, "gpt35_2nd_run")),
        # GPT-4o
        "gpt4o": ("gpt4o", os.path.join(repo_root, "gpt4o")),
        "gpt4o_2nd_run": ("gpt4o", os.path.join(repo_root, "gpt4o_2nd_run")),
        # DeepSeek (JSON keys are 'deepseek' and 'deepseek_2nd_run')
        "deepseek": ("deepseek", os.path.join(repo_root, "deep_seek")),
        "deepseek_2nd_run": ("deepseek", os.path.join(repo_root, "deep_seek_2nd_run")),
        # o1-mini (JSON keys use hyphen)
        "o1-mini": ("o1mini", os.path.join(repo_root, "o1_mini")),
        "o1-mini_2nd_run": ("o1mini", os.path.join(repo_root, "o1_mini_2nd_run")),
        # o3-mini
        "o3_mini_1st_run": ("o3mini", os.path.join(repo_root, "o3_mini_1st_run")),
        "o3_mini_2nd_run": ("o3mini", os.path.join(repo_root, "o3_mini_2nd_run")),
        # Claude (first run key in JSON is 'claude_3_7_sonnet')
        "claude_3_7_sonnet": (
            "claude",
            os.path.join(repo_root, "claude3_sonnet_1st_run"),
        ),
        "claude3_sonnet_2nd_run": (
            "claude",
            os.path.join(repo_root, "claude3_sonnet_2nd_run"),
        ),
    }

    out_root = os.path.join(repo_root, "LLM-WT")
    ensure_dir(out_root)

    total_copied = 0
    total_missing = 0

    # Ensure all LLM and run subdirectories exist regardless of copies
    for rk, (llm_name, _) in run_to_llm.items():
        llm_dir = os.path.join(out_root, llm_name)
        ensure_dir(llm_dir)
        ensure_dir(os.path.join(llm_dir, rk))

    for run_key, files in data.items():
        if run_key not in run_to_llm:
            continue

        llm_name, run_src_dir = run_to_llm[run_key]
        # Create per-LLM and per-run directories
        llm_dir = os.path.join(out_root, llm_name)
        ensure_dir(llm_dir)
        dest_dir = os.path.join(llm_dir, run_key)
        ensure_dir(dest_dir)

        for filename in files:
            if copy_from_run_recursive(run_src_dir, filename, dest_dir):
                total_copied += 1
            else:
                total_missing += 1

    print(f"Copied: {total_copied}")
    print(f"Missing: {total_missing}")
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()
