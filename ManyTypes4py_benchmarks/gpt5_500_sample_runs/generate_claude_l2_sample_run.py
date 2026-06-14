"""
Run Claude Opus 4.6 type annotation on the 500 sample files with L2 dependency context.

Based on generate_claude_sample_run.py, but reads from 500_untyped_files/ and
includes the matching dependency_slices/L2/*.capsule.pyi for each file.

Usage:
    python generate_claude_l2_sample_run.py           # all unprocessed files
    python generate_claude_l2_sample_run.py 4        # up to 4 unprocessed files
    python generate_claude_l2_sample_run.py 0        # no-op (limit <= 0)
"""

import json
import os
import sys
import time

import anthropic
import tiktoken
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PARENT_DIR)

load_dotenv(os.path.join(REPO_ROOT, ".env"))

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from environment

MODEL_NAME = "claude-opus-4-6"
RUN_NAME = "claude_opus_l2_run"

UNTYPED_DIR = os.path.join(PARENT_DIR, "500_untyped_files")
L2_SLICE_DIR = os.path.join(PARENT_DIR, "dependency_slices", "L2")
SELECTED_FILES_JSON = os.path.join(SCRIPT_DIR, "selected_500_files.json")
LOGS_DIR = os.path.join(PARENT_DIR, "Files_not_for_root_directories")

OUTPUT_DIR = os.path.join(PARENT_DIR, RUN_NAME)
PROCESSED_LOG = os.path.join(LOGS_DIR, f"processed_files_{RUN_NAME}.txt")
TIMING_LOG = os.path.join(LOGS_DIR, f"{RUN_NAME}_model_timings.json")
UNPROCESSED_LOG = os.path.join(LOGS_DIR, f"unprocessed_files_{RUN_NAME}.txt")
MISSING_SLICE_LOG = os.path.join(LOGS_DIR, f"missing_l2_slice_{RUN_NAME}.txt")
UNEXPECTED_FORMAT_LOG = os.path.join(LOGS_DIR, f"unexpected_format_{RUN_NAME}.txt")

SYSTEM_PROMPT = """You are a Python typing expert. You will receive:
1. A target Python module that needs type annotations.
2. An L2 dependency slice: a read-only .pyi-style summary of imported internal
   and third-party symbols (with types where available).

Your task: add type annotations to the TARGET MODULE only.

Rules:
1. Only add type annotations to function parameters, return types, and variables
   where appropriate. Do not change program logic or existing code structure.
2. Use the dependency slice to resolve imported symbol types. Prefer types from
   the slice when a symbol appears there; use Any only when truly unknown.
3. Do NOT reproduce, modify, or output the dependency slice.
4. Do not add explanations, comments, or extra prose.
5. Output only the complete annotated Python program inside a single ```python fence.

Use standard Python type hints (PEP 484) suitable for mypy."""


def get_token_count(text: str, model: str = MODEL_NAME) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def log_timing(filename: str, duration: float) -> None:
    if os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append({"file": filename, "time_taken": duration})
    with open(TIMING_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_processed_files() -> set[str]:
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def l2_slice_path(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return os.path.join(L2_SLICE_DIR, f"{stem}.capsule.pyi")


def _build_user_message(code: str, dependency_slice: str, module_name: str) -> str:
    return (
        f"Target module: {module_name!r}\n\n"
        f"<<<TARGET MODULE START>>>\n{code}\n<<<TARGET MODULE END>>>\n\n"
        f"L2 dependency slice (read-only context — do not output this):\n\n"
        f"<<<L2 DEPENDENCY SLICE START>>>\n"
        f"{dependency_slice}\n"
        f"<<<L2 DEPENDENCY SLICE END>>>"
    )


def generate_type_annotated_code(
    code: str, dependency_slice: str, module_name: str
) -> tuple[object, bool, float]:
    user_msg = _build_user_message(code, dependency_slice, module_name)
    prompt_tokens = get_token_count(SYSTEM_PROMPT + user_msg, model=MODEL_NAME)
    max_tokens = min(64000, (prompt_tokens * 2) + 1000)

    max_retries = 3
    wait_time = 60
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            content = ""
            with client.messages.stream(
                model=MODEL_NAME,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
                temperature=0,
            ) as stream:
                for text in stream.text_stream:
                    content += text
            duration = time.time() - start_time

            return content, True, duration
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return code, False, 0.0
            if "rate_limit_exceeded" in error_msg:
                print(
                    f"Rate limit. Retrying in {wait_time}s "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                wait_time += 30
                continue
            return code, False, 0.0

    print("Max retries reached. Skipping.")
    return code, False, 0.0


def resolve_limit(limit: int | None, unprocessed_count: int) -> int:
    """Return how many files to process this invocation."""
    if limit is None:
        return unprocessed_count
    if limit <= 0:
        return 0
    return min(limit, unprocessed_count)


def process_files(limit: int | None = None) -> None:
    with open(SELECTED_FILES_JSON, "r", encoding="utf-8") as f:
        selected_filenames = json.load(f)["files"]

    processed_files = load_processed_files()
    unprocessed = [f for f in selected_filenames if f not in processed_files]
    to_process = resolve_limit(limit, len(unprocessed))

    total = len(selected_filenames)
    already_done = total - len(unprocessed)
    print(
        f"Claude Opus L2: {total} selected files, "
        f"{already_done} already done, {len(unprocessed)} unprocessed, "
        f"{to_process} to process this run"
    )

    if to_process == 0:
        print("Nothing to process.")
        return

    files_to_run = unprocessed[:to_process]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed_count = 0

    for filename in files_to_run:
        untyped_path = os.path.join(UNTYPED_DIR, filename)
        slice_path = l2_slice_path(filename)

        if not os.path.isfile(untyped_path):
            print(f"Skipping {filename} — untyped file not found")
            with open(UNPROCESSED_LOG, "a", encoding="utf-8") as f:
                f.write(f"{filename}\tmissing_untyped\n")
            continue

        if not os.path.isfile(slice_path):
            print(f"Skipping {filename} — L2 slice not found: {slice_path}")
            with open(MISSING_SLICE_LOG, "a", encoding="utf-8") as f:
                f.write(filename + "\n")
            continue

        print(f"Processing: {filename}")
        try:
            with open(untyped_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            with open(slice_path, "r", encoding="utf-8", errors="ignore") as f:
                dependency_slice = f.read()
        except (UnicodeDecodeError, IOError) as e:
            print(f"Read error, skipping: {e}")
            continue

        module_name = os.path.splitext(filename)[0]
        modified_code, success, duration = generate_type_annotated_code(
            code, dependency_slice, module_name
        )
        log_timing(filename, duration)

        if not success:
            print(f"Skipping {filename} — generation failed")
            with open(UNPROCESSED_LOG, "a", encoding="utf-8") as f:
                f.write(f"{filename}\tgeneration_failed\n")
            continue

        content = (
            modified_code.content
            if hasattr(modified_code, "content")
            else modified_code
        )
        if isinstance(content, list):
            content = "".join(getattr(block, "text", str(block)) for block in content)

        try:
            code_block = content.split("```python\n")[1].split("```")[0]
        except IndexError:
            print(f"Skipping {filename} — unexpected format")
            with open(UNEXPECTED_FORMAT_LOG, "a", encoding="utf-8") as f:
                f.write(f"File: {filename}\nResponse:\n{content}\n{'=' * 40}\n")
            continue

        out_path = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(code_block)
        except (UnicodeEncodeError, IOError) as e:
            print(f"Write error, skipping: {e}")
            continue

        with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
            f.write(filename + "\n")

        processed_count += 1
        print(f"Done [{processed_count}/{to_process}]: {filename}")
        time.sleep(5)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print("Usage: python generate_claude_l2_sample_run.py [limit]")
            print("  limit: optional; <=0 skips, >0 caps this run (default: all unprocessed)")
            sys.exit(1)
    else:
        limit = None

    process_files(limit=limit)
