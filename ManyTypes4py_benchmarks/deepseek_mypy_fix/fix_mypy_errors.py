import os
import json
import subprocess
import glob
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
SOURCE_DIR = os.path.join(PARENT_DIR, "deep_seek_2nd_run")
FAILURE_JSON = os.path.join(
    PARENT_DIR, "mypy_results", "Section_04", "llm_only_failure_files.json"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "fixed_files")
LOG_FILE = os.path.join(BASE_DIR, "fix_log.json")
MAX_ATTEMPTS = 5


def find_file(filename):
    matches = glob.glob(os.path.join(SOURCE_DIR, "**", filename), recursive=True)
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


def fix_with_llm(code, mypy_errors):
    prompt = (
        "Here is a Python file with type annotations that has mypy errors.\n\n"
        "### Python Code:\n```python\n" + code + "\n```\n\n"
        "### Mypy Errors:\n```\n" + mypy_errors + "\n```\n\n"
        "Fix the type annotations so that mypy passes without errors. "
        "Keep all logic and functionality identical. "
        "Output ONLY the corrected Python code, no explanation."
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Python typing expert. Fix type annotation errors."},
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


def process_file(filename, log):
    src_path = find_file(filename)
    if not src_path:
        print(f"  SKIP: {filename} not found in {SOURCE_DIR}")
        log[filename] = {"status": "not_found", "attempts": 0, "time_seconds": 0}
        return

    work_path = os.path.join(OUTPUT_DIR, filename)
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

        print(f"  Attempt {attempt}/{MAX_ATTEMPTS} — mypy errors found, sending to LLM...")
        fixed_code, success = fix_with_llm(code, errors)
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FAILURE_JSON, "r") as f:
        failures = json.load(f)

    file_list = failures.get("deepseek_2nd_run", [])
    TEST_LIMIT = None  # Set to None to process all files
    if TEST_LIMIT:
        file_list = file_list[:TEST_LIMIT]
    print(f"Total files to fix: {len(file_list)}")

    log = {}
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, ValueError):
            log = {}

    for i, filename in enumerate(file_list, 1):
        if filename in log:
            print(f"[{i}/{len(file_list)}] {filename} — already processed, skipping")
            continue
        print(f"[{i}/{len(file_list)}] {filename}")
        process_file(filename, log)

        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    fixed = sum(1 for v in log.values() if v["status"] == "fixed")
    unfixed = sum(1 for v in log.values() if v["status"] == "unfixed")
    print(f"\nDone. Fixed: {fixed}, Unfixed: {unfixed}, Total: {len(log)}")


if __name__ == "__main__":
    main()
