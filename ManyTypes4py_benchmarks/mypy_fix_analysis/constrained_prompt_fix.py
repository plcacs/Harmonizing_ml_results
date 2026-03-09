"""
Q3: What if we require LLMs to NOT change code structure and explicitly
allow them to use Any for type annotations?

Same iterative mypy-fix loop, but with a constrained prompt.
Run with: python constrained_prompt_fix.py --model gpt5
          python constrained_prompt_fix.py --model deepseek
"""

import os
import json
import subprocess
import glob
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FAILURE_JSON = os.path.join(
    PARENT_DIR, "mypy_results", "Section_04", "llm_only_failure_files.json"
)
MAX_ATTEMPTS = 5

MODEL_CONFIGS = {
    "gpt5": {
        "client_kwargs": {"api_key": os.getenv("OPENAI_API_KEY")},
        "model_name": "gpt-5",
        "source_dir": os.path.join(PARENT_DIR, "gpt5_1st_run"),
        "failure_key": "gpt5_1st_run",
        "output_dir": os.path.join(BASE_DIR, "constrained_fixed_files", "gpt5"),
        "log_file": os.path.join(BASE_DIR, "constrained_fix_log_gpt5.json"),
    },
    "deepseek": {
        "client_kwargs": {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com",
        },
        "model_name": "deepseek-chat",
        "source_dir": os.path.join(PARENT_DIR, "deep_seek_2nd_run"),
        "failure_key": "deepseek_2nd_run",
        "output_dir": os.path.join(BASE_DIR, "constrained_fixed_files", "deepseek"),
        "log_file": os.path.join(BASE_DIR, "constrained_fix_log_deepseek.json"),
    },
}

CONSTRAINED_PROMPT = (
    "Here is a Python file with type annotations that has mypy errors.\n\n"
    "### Python Code:\n```python\n{code}\n```\n\n"
    "### Mypy Errors:\n```\n{errors}\n```\n\n"
    "Fix ONLY the type annotations to resolve the mypy errors. You MUST follow these rules:\n"
    "1. Do NOT change the code structure, logic, control flow, variable names, or function bodies.\n"
    "2. Do NOT add or remove functions, classes, or methods.\n"
    "3. Do NOT add or remove parameters from functions.\n"
    "4. You MAY change any type annotation to 'Any' if you cannot determine the correct type.\n"
    "5. You MAY add typing imports (e.g., from typing import Any, cast).\n"
    "6. You MAY add '# type: ignore' comments as a last resort.\n\n"
    "Output ONLY the corrected Python code, no explanation."
)


def find_file(directory, filename):
    matches = glob.glob(os.path.join(directory, "**", filename), recursive=True)
    return matches[0] if matches else None


def run_mypy(filepath):
    abs_path = os.path.abspath(filepath)
    result = subprocess.run(
        [
            "mypy",
            "--ignore-missing-imports",
            "--allow-untyped-defs",
            "--no-incremental",
            "--python-version=3.10",
            "--disable-error-code=no-redef",
            "--cache-dir=/dev/null",
            abs_path,
        ],
        cwd=os.path.dirname(abs_path),
        capture_output=True,
        text=True,
        check=False,
    )
    errors = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, errors


def fix_with_llm(client, model_name, code, mypy_errors):
    prompt = CONSTRAINED_PROMPT.format(code=code, errors=mypy_errors)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a Python typing expert. Fix type annotation errors without changing code structure."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return content, True
    except Exception as e:
        print(f"  LLM error: {e}")
        return code, False


def process_file(client, model_name, filename, source_dir, output_dir, log):
    src_path = find_file(source_dir, filename)
    if not src_path:
        print(f"  SKIP: {filename} not found")
        log[filename] = {"status": "not_found", "attempts": 0, "time_seconds": 0}
        return

    work_path = os.path.join(output_dir, filename)
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    with open(work_path, "w", encoding="utf-8") as f:
        f.write(code)

    start_time = time.time()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        passed, errors = run_mypy(work_path)
        if passed:
            elapsed = round(time.time() - start_time, 2)
            print(f"  PASS on attempt {attempt} ({elapsed}s)")
            log[filename] = {"status": "fixed", "attempts": attempt, "time_seconds": elapsed}
            return

        print(f"  Attempt {attempt}/{MAX_ATTEMPTS} — sending to LLM (constrained)...")
        fixed_code, success = fix_with_llm(client, model_name, code, errors)
        if not success:
            elapsed = round(time.time() - start_time, 2)
            log[filename] = {"status": "llm_error", "attempts": attempt, "time_seconds": elapsed}
            return

        with open(work_path, "w", encoding="utf-8") as f:
            f.write(fixed_code)
        code = fixed_code
        time.sleep(2)

    passed, _ = run_mypy(work_path)
    elapsed = round(time.time() - start_time, 2)
    status = "fixed" if passed else "unfixed"
    print(f"  Final status: {status} ({elapsed}s)")
    log[filename] = {"status": status, "attempts": MAX_ATTEMPTS, "time_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt5", "deepseek"], required=True)
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    client = OpenAI(**config["client_kwargs"])
    os.makedirs(config["output_dir"], exist_ok=True)

    with open(FAILURE_JSON, "r") as f:
        failures = json.load(f)

    file_list = failures.get(config["failure_key"], [])
    TEST_LIMIT = 3  # Set to None to process all files
    if TEST_LIMIT:
        file_list = file_list[:TEST_LIMIT]
    print(f"Total files to fix ({args.model}, constrained prompt): {len(file_list)}")

    log = {}
    if os.path.exists(config["log_file"]):
        try:
            with open(config["log_file"], "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, ValueError):
            log = {}

    for i, filename in enumerate(file_list, 1):
        if filename in log:
            print(f"[{i}/{len(file_list)}] {filename} — already processed, skipping")
            continue
        print(f"[{i}/{len(file_list)}] {filename}")
        process_file(client, config["model_name"], filename, config["source_dir"], config["output_dir"], log)

        with open(config["log_file"], "w") as f:
            json.dump(log, f, indent=2)

    fixed = sum(1 for v in log.values() if v["status"] == "fixed")
    unfixed = sum(1 for v in log.values() if v["status"] == "unfixed")
    print(f"\nDone. Fixed: {fixed}, Unfixed: {unfixed}, Total: {len(log)}")


if __name__ == "__main__":
    main()
