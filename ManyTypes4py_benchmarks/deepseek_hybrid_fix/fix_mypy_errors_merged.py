"""
Option 2: Merged-file iterative mypy error fixer.

Pipeline per file (up to MAX_ATTEMPTS iterations):
  1. Send the merged .py file (with annotations) + mypy errors to LLM
  2. LLM returns the fixed merged file (only annotation changes allowed)
  3. Re-run mypy on the fixed file
  4. If errors remain, send the updated file + new errors (multi-turn conversation)
"""

import ast
import os
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

MERGED_DIR = os.path.join(PARENT_DIR, "deepseek_3_stub_run", "merged")
MYPY_RESULTS_JSON = os.path.join(
    PARENT_DIR,
    "GPCE_mypy_results",
    "mypy_outputs",
    "mypy_results_deepseek_3_merge_stub_run_with_errors.json",
)

OUTPUT_DIR = os.path.join(BASE_DIR, "fixed_merged_option2")
LOG_FILE = os.path.join(BASE_DIR, "fix_log_option2.json")

MAX_ATTEMPTS = 5


SYSTEM_PROMPT = (
    "You are a Python typing expert.\n"
    "You will be given a Python file with type annotations and mypy errors.\n\n"
    "STRICT EDIT RULES (must follow):\n"
    "1) You may modify ONLY:\n"
    "   - type annotations (function parameters, returns, variable annotations, class attribute annotations, TypeAlias)\n"
    "   - typing-related imports (from typing / typing_extensions)\n"
    "   - `if TYPE_CHECKING:` import blocks\n"
    "2) Do NOT change runtime behavior:\n"
    "   - Do not change executable code, control flow, expressions, values, string literals, or function bodies.\n"
    "   - Do not rename symbols, reorder code, reformat, or add/remove non-typing statements.\n"
    "3) Do NOT add `# type: ignore` comments. Do NOT use cast.\n"
    "4) You may remove type annotations or use Any but prefer precise typing.\n"
    "5) Keep the file complete and syntactically valid. Preserve all existing code outside the allowed edits.\n\n"
    "Output the entire fixed file as plain text only. No explanations, no markdown fences.\n"
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


def build_first_turn_prompt(code, mypy_errors):
    return (
        "### Python File:\n"
        f"```python\n{code}\n```\n\n"
        "### Mypy Errors:\n"
        f"```\n{mypy_errors}\n```\n\n"
        "Fix the type annotations to resolve these mypy errors. "
        "Output the entire corrected file only."
    )


def build_followup_prompt(code, mypy_errors):
    return (
        "The previous fix still has mypy errors. Here is the updated state:\n\n"
        "### Current File:\n"
        f"```python\n{code}\n```\n\n"
        "### Remaining Mypy Errors:\n"
        f"```\n{mypy_errors}\n```\n\n"
        "Fix the type annotations. Output the entire corrected file only."
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
    merged_path = os.path.join(MERGED_DIR, filename)

    if not os.path.exists(merged_path):
        print(f"  SKIP: merged file not found: {merged_path}")
        log[filename] = {"status": "not_found", "attempts": 0, "time_seconds": 0}
        return

    with open(merged_path, "r", encoding="utf-8", errors="ignore") as f:
        current_code = f.read()

    initial_error_count = initial_errors_info.get("error_count", 0)
    initial_error_messages = initial_errors_info.get("errors", [])
    mypy_errors = "\n".join(initial_error_messages)

    work_path = os.path.join(OUTPUT_DIR, filename)
    with open(work_path, "w", encoding="utf-8") as f:
        f.write(current_code)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    start_time = time.time()
    iteration_log = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt == 1:
            user_msg = build_first_turn_prompt(current_code, mypy_errors)
        else:
            user_msg = build_followup_prompt(current_code, mypy_errors)

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

        fixed_code = strip_markdown_fences(reply.strip())

        # Validate syntax
        try:
            ast.parse(fixed_code)
        except SyntaxError:
            print(f"  Attempt {attempt}: LLM returned invalid syntax, retrying")
            iteration_log.append({"attempt": attempt, "error_count": -1, "note": "invalid_syntax"})
            mypy_errors = "The file you returned has a SyntaxError. Please fix it."
            time.sleep(2)
            continue

        with open(work_path, "w", encoding="utf-8") as f:
            f.write(fixed_code)

        # Re-run mypy
        passed, errors = run_mypy_on_file(work_path)
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

        current_code = fixed_code
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

    parser = argparse.ArgumentParser(description="Merged-file iterative mypy error fixer (Option 2)")
    parser.add_argument(
        "-n", "--limit", type=int, default=None,
        help="Number of files to process (default: all). Use 10 for a trial run.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from where the last run left off (skip already-processed files).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
