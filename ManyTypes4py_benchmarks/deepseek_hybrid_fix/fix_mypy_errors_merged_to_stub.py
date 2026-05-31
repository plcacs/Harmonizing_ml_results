"""
Option 3: Send merged file + mypy errors → receive fixed .pyi stub → re-merge with untyped → re-run mypy.

Pipeline per file (up to MAX_ATTEMPTS iterations):
  1. Send the merged .py file (with annotations) + mypy errors to LLM
  2. LLM returns a fixed .pyi stub file
  3. Re-merge untyped .py + fixed .pyi → merged .py
  4. Re-run mypy on the merged file
  5. If errors remain, send the new merged file + new errors (multi-turn conversation)
"""

import ast
import os
import sys
import json
import subprocess
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

UNTYPED_DIR = os.path.join(PARENT_DIR, "500_untyped_files")
MERGED_DIR = os.path.join(PARENT_DIR, "deepseek_3_stub_run", "merged")
MYPY_RESULTS_JSON = os.path.join(
    PARENT_DIR,
    "HarmonizingML_mypy_results",
    "mypy_outputs",
    "mypy_results_deepseek_3_merge_stub_run_with_errors.json",
)

OUTPUT_STUB_DIR = os.path.join(BASE_DIR, "fixed_stubs_option3")
OUTPUT_MERGED_DIR = os.path.join(BASE_DIR, "fixed_merged_option3")
LOG_FILE = os.path.join(BASE_DIR, "fix_log_option3.json")

MAX_ATTEMPTS = 5

sys.path.insert(0, os.path.join(PARENT_DIR, "HarmonizingML_Type_info_analysis"))
from merge_stubs_into_py import merge_stub_into_py


SYSTEM_PROMPT = (
    "You are a Python typing expert specializing in .pyi stub files.\n"
    "You will be given a merged Python file (source code with type annotations inlined) "
    "and mypy errors produced from that file.\n\n"
    "Your task: produce a .pyi stub file that, when its annotations are merged back into "
    "the original untyped source, will pass mypy without errors.\n\n"
    "Rules:\n"
    "1. Output ONLY the complete .pyi stub file. No explanations, no markdown fences.\n"
    "2. The stub must contain function signatures, class definitions, and variable "
    "annotations that match the structure of the given file.\n"
    "3. Use precise types. You may use TypeVar, Protocol, TypedDict, Callable, overloads, etc.\n"
    "4. You may leave parameters/returns unannotated if the correct type is too complex.\n"
    "5. Do NOT add `# type: ignore` comments.\n"
    "6. Keep the stub syntactically valid.\n"
    "7. Every function/class in the source should appear in the stub.\n"
)


def run_mypy_on_file(filepath):
    abs_path = os.path.abspath(filepath)
    result = subprocess.run(
        [
            "mypy",
            "--ignore-missing-imports",
            "--allow-untyped-defs",
            "--no-incremental",
            "--python-version=3.10",
            "--disable-error-code=no-redef",
            "--cache-dir=nul",
            abs_path,
        ],
        cwd=os.path.dirname(abs_path),
        capture_output=True,
        text=True,
        check=False,
    )
    errors = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, errors


def strip_markdown_fences(text):
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def build_first_turn_prompt(merged_code, mypy_errors):
    return (
        "### Merged Python File (source + annotations):\n"
        f"```python\n{merged_code}\n```\n\n"
        "### Mypy Errors:\n"
        f"```\n{mypy_errors}\n```\n\n"
        "Produce a fixed .pyi stub file that resolves these mypy errors. "
        "Output the complete .pyi stub only."
    )


def build_followup_prompt(merged_code, mypy_errors):
    return (
        "The previous stub still produces mypy errors after merging. "
        "Here is the updated state:\n\n"
        "### Current Merged File (after re-merge with your stub):\n"
        f"```python\n{merged_code}\n```\n\n"
        "### Remaining Mypy Errors:\n"
        f"```\n{mypy_errors}\n```\n\n"
        "Output the fixed .pyi stub file only."
    )


def get_files_with_errors(mypy_json_path):
    with open(mypy_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        fname: info
        for fname, info in data.items()
        if info.get("error_count", 0) > 0
    }


def process_file(filename, initial_errors_info, log):
    stem = os.path.splitext(filename)[0]
    untyped_path = os.path.join(UNTYPED_DIR, filename)
    merged_path_src = os.path.join(MERGED_DIR, filename)

    if not os.path.exists(untyped_path):
        print(f"  SKIP: untyped file not found: {untyped_path}")
        log[filename] = {"status": "not_found", "attempts": 0, "time_seconds": 0}
        return

    if not os.path.exists(merged_path_src):
        print(f"  SKIP: merged file not found: {merged_path_src}")
        log[filename] = {"status": "merged_not_found", "attempts": 0, "time_seconds": 0}
        return

    with open(merged_path_src, "r", encoding="utf-8", errors="ignore") as f:
        current_merged = f.read()

    initial_error_count = initial_errors_info.get("error_count", 0)
    initial_error_messages = initial_errors_info.get("errors", [])
    mypy_errors = "\n".join(initial_error_messages)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    start_time = time.time()
    iteration_log = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt == 1:
            user_msg = build_first_turn_prompt(current_merged, mypy_errors)
        else:
            user_msg = build_followup_prompt(current_merged, mypy_errors)

        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"  LLM error on attempt {attempt}: {e}")
            elapsed = round(time.time() - start_time, 2)
            log[filename] = {
                "status": "llm_error",
                "attempts": attempt,
                "initial_errors": initial_error_count,
                "initial_error_messages": initial_error_messages,
                "final_error_messages": mypy_errors.splitlines() if mypy_errors else [],
                "iterations": iteration_log,
                "time_seconds": elapsed,
            }
            return

        fixed_stub = strip_markdown_fences(reply.strip())

        try:
            ast.parse(fixed_stub)
        except SyntaxError:
            print(f"  Attempt {attempt}: LLM returned invalid stub syntax, retrying")
            iteration_log.append({"attempt": attempt, "error_count": -1, "note": "invalid_stub_syntax"})
            mypy_errors = "The .pyi stub you returned has a SyntaxError. Please fix it."
            time.sleep(2)
            continue

        out_stub_path = os.path.join(OUTPUT_STUB_DIR, stem + ".pyi")
        with open(out_stub_path, "w", encoding="utf-8") as f:
            f.write(fixed_stub)

        merged_source = merge_stub_into_py(untyped_path, out_stub_path)
        if merged_source is None:
            print(f"  Attempt {attempt}: merge failed")
            iteration_log.append({"attempt": attempt, "error_count": -1, "note": "merge_failed"})
            mypy_errors = (
                "The .pyi stub could not be merged with the untyped source. "
                "Ensure function/class signatures match the source structure. Please fix."
            )
            time.sleep(2)
            continue

        out_merged_path = os.path.join(OUTPUT_MERGED_DIR, filename)
        with open(out_merged_path, "w", encoding="utf-8") as f:
            f.write(merged_source)

        passed, errors = run_mypy_on_file(out_merged_path)
        error_count = 0 if passed else errors.count("\n")

        iteration_log.append({"attempt": attempt, "error_count": error_count})
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS} — errors: {error_count}")

        if passed:
            elapsed = round(time.time() - start_time, 2)
            print(f"  PASS on attempt {attempt} ({elapsed}s)")
            log[filename] = {
                "status": "fixed",
                "attempts": attempt,
                "initial_errors": initial_error_count,
                "initial_error_messages": initial_error_messages,
                "final_error_messages": [],
                "iterations": iteration_log,
                "time_seconds": elapsed,
            }
            return

        current_merged = merged_source
        mypy_errors = errors
        time.sleep(2)

    elapsed = round(time.time() - start_time, 2)
    print(f"  UNFIXED after {MAX_ATTEMPTS} attempts ({elapsed}s)")
    log[filename] = {
        "status": "unfixed",
        "attempts": MAX_ATTEMPTS,
        "initial_errors": initial_error_count,
        "initial_error_messages": initial_error_messages,
        "final_error_messages": mypy_errors.splitlines() if mypy_errors else [],
        "iterations": iteration_log,
        "time_seconds": elapsed,
    }


def print_summary(log):
    fixed = sum(1 for v in log.values() if v["status"] == "fixed")
    unfixed = sum(1 for v in log.values() if v["status"] == "unfixed")
    llm_err = sum(1 for v in log.values() if v["status"] == "llm_error")
    other = len(log) - fixed - unfixed - llm_err
    print(f"\n{'='*60}")
    print(f"Summary: Fixed={fixed}, Unfixed={unfixed}, LLM_Error={llm_err}, Other={other}, Total={len(log)}")
    print(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merged-to-stub iterative mypy error fixer (Option 3)"
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None,
        help="Number of files to process (default: all). Use 10 for a trial run.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from where the last run left off (skip already-processed files).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_STUB_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MERGED_DIR, exist_ok=True)

    error_files = get_files_with_errors(MYPY_RESULTS_JSON)
    file_list = sorted(error_files.keys())

    log = {}
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, ValueError):
            log = {}

    remaining = [f for f in file_list if f not in log]

    if args.limit:
        remaining = remaining[: args.limit]

    print(f"Files with errors: {len(file_list)} | Already processed: {len(log)} | This run: {len(remaining)}")

    for i, filename in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {filename}")
        process_file(filename, error_files[filename], log)

        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    print_summary(log)


if __name__ == "__main__":
    main()
