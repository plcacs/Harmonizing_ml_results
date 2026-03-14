import os
import sys
import json
import subprocess
import time
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
UNTYPED_DIR = os.path.join(PARENT_DIR, "untyped_benchmarks")
UNPROCESSED_JSON = os.path.join(
    PARENT_DIR, "mypy_results", "Section_04", "unprocessed_files.json"
)
MAX_ATTEMPTS = 5

MODEL_CONFIGS = {
    "gpt5": {
        "json_key": "gpt5_1st_run",
        "api": "openai",
        "model": "gpt-5",
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
    },
    "claude": {
        "json_key": "claude_3_7_sonnet",
        "api": "anthropic",
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    "deepseek": {
        "json_key": "deepseek_2nd_run",
        "api": "openai",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
}

NON_TYPE_ERROR_CODES = [
    "name-defined",
    "import",
    "syntax",
    "no-redef",
    "unused-ignore",
    "override-without-super",
    "redundant-cast",
    "literal-required",
    "typeddict-unknown-key",
    "typeddict-item",
    "truthy-function",
    "str-bytes-safe",
    "unused-coroutine",
    "explicit-override",
    "truthy-iterable",
    "redundant-self",
    "redundant-await",
    "unreachable",
]


def has_non_type_error(mypy_output):
    """Check if mypy output contains non-type-related errors (syntax, import, etc.)."""
    errors = [line for line in mypy_output.splitlines() if line.strip()]

    for error in errors:
        lower = error.lower()
        if any(kw in lower for kw in ["syntax", "empty_body", "name_defined"]):
            return True
        if "[" in error and "]" in error:
            code = error[error.rindex("[") + 1 : error.rindex("]")]
            if code in NON_TYPE_ERROR_CODES:
                return True
    return False


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
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output


def strip_code_fences(content):
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    return content


def create_client(config):
    if config["api"] == "anthropic":
        import anthropic

        return anthropic.Anthropic()

    from openai import OpenAI

    kwargs = {"api_key": os.getenv(config["env_key"])}
    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]
    return OpenAI(**kwargs)


def call_llm(client, config, prompt, system_msg):
    if config["api"] == "anthropic":
        content = ""
        with client.messages.stream(
            model=config["model"],
            max_tokens=64000,
            messages=[{"role": "user", "content": prompt}],
            system=system_msg,
            temperature=0,
        ) as stream:
            for text in stream.text_stream:
                content += text
        return strip_code_fences(content)

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    return strip_code_fences(response.choices[0].message.content)


def generate_initial(client, config, original_code):
    prompt = (
        f"Here is a Python program:\n\n{original_code}\n\n"
        "Add appropriate type annotations. "
        "Do not change function signatures, imports, or class definitions. "
        "Output only the annotated Python code. No explanation needed."
    )
    return call_llm(
        client, config, prompt, "You are a Python programming expert."
    )


def regenerate_with_feedback(client, config, original_code, previous_errors):
    prompt = (
        "Here is a Python program:\n\n"
        f"```python\n{original_code}\n```\n\n"
        "A previous attempt to add type annotations introduced these errors:\n"
        f"```\n{previous_errors}\n```\n\n"
        "Add appropriate type annotations to the original code above. "
        "Do NOT introduce syntax errors, import errors, or undefined names. "
        "Keep all logic, imports, and class/function definitions identical. "
        "Output ONLY the corrected annotated Python code, no explanation."
    )
    return call_llm(
        client, config, prompt, "You are a Python typing expert."
    )


def move_to_status_dir(work_path, output_dir, filename, status):
    """Move the generated file into a status-based subdirectory."""
    status_dir = os.path.join(output_dir, status)
    os.makedirs(status_dir, exist_ok=True)
    dest_path = os.path.join(status_dir, filename)
    if os.path.exists(work_path):
        with open(work_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.remove(work_path)


def process_file(filename, client, config, output_dir, log):
    original_path = os.path.join(UNTYPED_DIR, filename)
    if not os.path.exists(original_path):
        print(f"  SKIP: {filename} not found in untyped_benchmarks")
        log[filename] = {"status": "not_found", "attempts": 0, "time_seconds": 0}
        return

    with open(original_path, "r", encoding="utf-8", errors="ignore") as f:
        original_code = f.read()

    work_path = os.path.join(output_dir, filename)
    start_time = time.time()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if attempt == 1:
                generated = generate_initial(client, config, original_code)
            else:
                generated = regenerate_with_feedback(
                    client, config, original_code, last_errors
                )
        except Exception as e:
            print(f"  LLM error on attempt {attempt}: {e}")
            elapsed = round(time.time() - start_time, 2)
            log[filename] = {
                "status": "llm_error",
                "attempts": attempt,
                "time_seconds": elapsed,
            }
            return

        with open(work_path, "w", encoding="utf-8") as f:
            f.write(generated)

        passed, mypy_output = run_mypy(work_path)

        if passed:
            elapsed = round(time.time() - start_time, 2)
            print(f"  PASS (no errors) on attempt {attempt} ({elapsed}s)")
            move_to_status_dir(work_path, output_dir, filename, "clean")
            log[filename] = {
                "status": "clean",
                "attempts": attempt,
                "time_seconds": elapsed,
            }
            return

        if not has_non_type_error(mypy_output):
            elapsed = round(time.time() - start_time, 2)
            print(
                f"  OK (type-only errors) on attempt {attempt} ({elapsed}s)"
            )
            move_to_status_dir(work_path, output_dir, filename, "type_errors_only")
            log[filename] = {
                "status": "type_errors_only",
                "attempts": attempt,
                "time_seconds": elapsed,
            }
            return

        print(
            f"  Attempt {attempt}/{MAX_ATTEMPTS} — non-type errors found, retrying..."
        )
        last_errors = mypy_output
        time.sleep(2)

    # Final check after all attempts exhausted
    passed, mypy_output = run_mypy(work_path)
    has_structural = has_non_type_error(mypy_output) if not passed else False
    elapsed = round(time.time() - start_time, 2)
    status = "clean" if passed else ("type_errors_only" if not has_structural else "still_unprocessed")
    print(f"  Final status: {status} ({elapsed}s)")
    move_to_status_dir(work_path, output_dir, filename, status)
    log[filename] = {
        "status": status,
        "attempts": MAX_ATTEMPTS,
        "time_seconds": elapsed,
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODEL_CONFIGS:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(MODEL_CONFIGS.keys())}>")
        sys.exit(1)

    model_name = sys.argv[1]
    config = MODEL_CONFIGS[model_name]
    print(f"Model: {model_name} ({config['model']})")

    output_dir = os.path.join(BASE_DIR, f"regenerated_{model_name}")
    log_file = os.path.join(BASE_DIR, f"reprocess_log_{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(UNPROCESSED_JSON, "r") as f:
        all_unprocessed = json.load(f)

    file_list = all_unprocessed.get(config["json_key"], [])
    TEST_LIMIT = None  # Set to None to process all files
    if TEST_LIMIT:
        file_list = file_list[:TEST_LIMIT]
    print(f"Total unprocessed files: {len(file_list)}")

    client = create_client(config)

    log = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, ValueError):
            log = {}

    for i, filename in enumerate(file_list, 1):
        if filename in log:
            print(f"[{i}/{len(file_list)}] {filename} — already processed, skipping")
            continue
        print(f"[{i}/{len(file_list)}] {filename}")
        process_file(filename, client, config, output_dir, log)

        with open(log_file, "w") as f:
            json.dump(log, f, indent=2)

    clean = sum(1 for v in log.values() if v["status"] == "clean")
    type_only = sum(1 for v in log.values() if v["status"] == "type_errors_only")
    still_bad = sum(1 for v in log.values() if v["status"] == "still_unprocessed")
    errors = sum(1 for v in log.values() if v["status"] == "llm_error")
    print(
        f"\nDone. Clean: {clean}, Type-only: {type_only}, "
        f"Still unprocessed: {still_bad}, LLM errors: {errors}, Total: {len(log)}"
    )


if __name__ == "__main__":
    main()
