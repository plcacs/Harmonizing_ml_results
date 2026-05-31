"""
Run Claude to generate .pyi stub files for the 500 selected sample files.

Usage:
    python generate_claude_stub_sample_run.py <run_number> [max_files]
    python generate_claude_stub_sample_run.py <run_number> --batch          # submit batch
    python generate_claude_stub_sample_run.py <run_number> --retrieve <id>  # retrieve batch

    run_number: 1, 2, or 3
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

SYSTEM_PROMPT = """You are generating a Python type stub (.pyi) file for static type checking.

Produce a .pyi stub for the given module that follows these rules:

1. Preserve all public API:
   - Include all top-level functions, classes, methods, and public variables that appear in the module.
   - Keep the same names and parameter structures as in the original code.
2. Use standard Python type hints (PEP 484) suitable for mypy and other type checkers.
3. For any type you are unsure about, use 'Any' rather than guessing unsafely.
4. Use proper .pyi stub syntax:
   - Function and method bodies must be '...'.
   - Class bodies may contain method/attribute declarations with '...'.
   - Module-level variables should be annotated with a type and assigned '...'.
5. Do NOT include any executable code, logic, or imports that are only used at runtime.
6. Do NOT add explanations, comments, or extra prose; output only stub code.
7. The output must be a single, complete .pyi file corresponding to this module.

Return only valid .pyi stub code."""

SELECTED_FILES_JSON = os.path.join(SCRIPT_DIR, "selected_500_files.json")
GROUPED_JSON = os.path.join(
    PARENT_DIR, "Files_not_for_root_directories", "grouped_file_paths.json"
)


def _repo_rel_path(file_path: str) -> str:
    """grouped_file_paths.json uses '\\' separators; normalize for POSIX basename/join."""
    return file_path.replace("\\", "/")


def get_run_paths(run_number: int) -> dict:
    run_name = f"claude_stub_{run_number}_run"
    output_dir = os.path.join(PARENT_DIR, run_name)
    logs_dir = os.path.join(PARENT_DIR, "Files_not_for_root_directories")
    return {
        "output_dir": output_dir,
        "processed_log": os.path.join(output_dir, "processed_files.txt"),
        "timing_log": os.path.join(logs_dir, f"{run_name}_model_timings.json"),
        "unprocessed_log": os.path.join(logs_dir, f"unprocessed_files_{run_name}.txt"),
        "unexpected_format_log": os.path.join(
            logs_dir, f"unexpected_format_{run_name}.txt"
        ),
    }


def get_token_count(text: str, model: str = MODEL_NAME) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def log_timing(file_path: str, duration: float, timing_log: str) -> None:
    if os.path.exists(timing_log):
        with open(timing_log, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append({"file": file_path, "time_taken": duration})
    with open(timing_log, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_processed_files(log_path: str) -> set[str]:
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def _build_user_message(code: str, module_name: str) -> str:
    return (
        f"Here is the implementation of a Python module named {module_name!r}:\n\n"
        f"<<<PYTHON MODULE START>>>\n{code}\n<<<PYTHON MODULE END>>>"
    )


def generate_stub(code: str, module_name: str) -> tuple[str, bool, float]:
    user_msg = _build_user_message(code, module_name)
    prompt_tokens = get_token_count(SYSTEM_PROMPT + user_msg, model=MODEL_NAME)
    max_tokens = min(64000, max(1024, prompt_tokens))

    max_retries = 3
    wait_time = 60
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.messages.create(
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
                stream=False,
            )
            duration = time.time() - start_time

            content = ""
            if hasattr(response, "content") and response.content:
                if isinstance(response.content, list):
                    content = "".join(
                        getattr(block, "text", str(block)) for block in response.content
                    )
                else:
                    content = str(response.content)
            else:
                content = str(response)

            return content, True, duration
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit — not retrying.")
                return "", False, 0.0
            if "rate_limit_exceeded" in error_msg:
                print(
                    f"Rate limit. Retrying in {wait_time}s "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                wait_time += 30
                continue
            return "", False, 0.0

    print("Max retries reached. Skipping.")
    return "", False, 0.0


def process_files(run_number: int, max_files: int | None = None) -> None:
    paths = get_run_paths(run_number)

    with open(SELECTED_FILES_JSON, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
    selected_filenames = set(selected_data["files"])

    with open(GROUPED_JSON, "r", encoding="utf-8") as f:
        file_map = json.load(f)

    processed_files = load_processed_files(paths["processed_log"])

    files_to_run: list[tuple[str, str]] = []
    for group_id in sorted(file_map.keys(), key=int):
        for file_path in file_map[group_id]:
            basename = os.path.basename(_repo_rel_path(file_path))
            if basename in selected_filenames:
                files_to_run.append((group_id, file_path))

    if max_files is not None:
        files_to_run = files_to_run[:max_files]

    total = len(files_to_run)
    already_done = sum(1 for _, fp in files_to_run if fp in processed_files)
    remaining = total - already_done
    print(
        f"Claude stub run {run_number}: {total} selected files, "
        f"{already_done} already done, {remaining} remaining"
    )

    processed_count = 0
    for _, file_path in files_to_run:
        if file_path in processed_files:
            continue

        print(f"Processing for stub: {file_path}")
        full_path = os.path.join(PARENT_DIR, _repo_rel_path(file_path))
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except (UnicodeDecodeError, IOError) as e:
            print(f"Read error, skipping: {e}")
            continue

        module_name = os.path.splitext(os.path.basename(_repo_rel_path(file_path)))[0]
        stub_content, success, duration = generate_stub(code, module_name)
        log_timing(file_path, duration, paths["timing_log"])

        if not success or not stub_content.strip():
            print(f"Skipping {file_path} — stub generation failed")
            with open(paths["unprocessed_log"], "a", encoding="utf-8") as f:
                f.write(file_path + "\n")
            continue

        os.makedirs(paths["output_dir"], exist_ok=True)

        base_name = os.path.splitext(os.path.basename(_repo_rel_path(file_path)))[0] + ".pyi"
        out_path = os.path.join(paths["output_dir"], base_name)
        try:
            with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(stub_content)
        except (UnicodeEncodeError, IOError) as e:
            print(f"Write error, skipping: {e}")
            continue

        with open(paths["processed_log"], "a", encoding="utf-8") as f:
            f.write(file_path + "\n")

        processed_count += 1
        print(f"Done stub [{processed_count}/{remaining}]: {file_path}")
        time.sleep(5)


def _collect_files(run_number: int, max_files: int | None = None):
    """Return (paths_dict, list_of_(group_id, file_path), processed_set)."""
    paths = get_run_paths(run_number)
    with open(SELECTED_FILES_JSON, "r", encoding="utf-8") as f:
        selected_filenames = set(json.load(f)["files"])
    with open(GROUPED_JSON, "r", encoding="utf-8") as f:
        file_map = json.load(f)

    processed = load_processed_files(paths["processed_log"])
    files_to_run = []
    for gid in sorted(file_map.keys(), key=int):
        for fp in file_map[gid]:
            if os.path.basename(_repo_rel_path(fp)) in selected_filenames:
                files_to_run.append((gid, fp))
    if max_files is not None:
        files_to_run = files_to_run[:max_files]
    return paths, files_to_run, processed


def submit_batch(run_number: int, max_files: int | None = None) -> None:
    """Build and submit a Message Batch (50% cost reduction)."""
    paths, files_to_run, processed = _collect_files(run_number, max_files)
    requests = []
    for _, file_path in files_to_run:
        if file_path in processed:
            continue
        full_path = os.path.join(PARENT_DIR, _repo_rel_path(file_path))
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except (UnicodeDecodeError, IOError):
            continue

        module_name = os.path.splitext(os.path.basename(_repo_rel_path(file_path)))[0]
        user_msg = _build_user_message(code, module_name)
        prompt_tokens = get_token_count(SYSTEM_PROMPT + user_msg, model=MODEL_NAME)

        requests.append({
            "custom_id": file_path,
            "params": {
                "model": MODEL_NAME,
                "max_tokens": min(64000, max(1024, prompt_tokens)),
                "temperature": 0,
                "system": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
        })

    if not requests:
        print("No files to process.")
        return

    print(f"Submitting batch with {len(requests)} requests …")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted!  id={batch.id}  status={batch.processing_status}")
    print(f"Retrieve results with:\n  python {sys.argv[0]} {run_number} --retrieve {batch.id}")


def retrieve_batch(run_number: int, batch_id: str) -> None:
    """Download completed batch results and write stubs to disk."""
    paths = get_run_paths(run_number)
    os.makedirs(paths["output_dir"], exist_ok=True)

    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch {batch_id}  status={batch.processing_status}")
    if batch.processing_status != "ended":
        print("Batch not finished yet. Try again later.")
        return

    saved = 0
    for result in client.messages.batches.results(batch_id):
        file_path = result.custom_id
        if result.result.type != "succeeded":
            print(f"FAILED: {file_path} — {result.result.type}")
            with open(paths["unprocessed_log"], "a", encoding="utf-8") as f:
                f.write(file_path + "\n")
            continue

        msg = result.result.message
        content = "".join(
            getattr(b, "text", str(b)) for b in msg.content
        ) if isinstance(msg.content, list) else str(msg.content)

        base_name = os.path.splitext(os.path.basename(_repo_rel_path(file_path)))[0] + ".pyi"
        out_path = os.path.join(paths["output_dir"], base_name)
        with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(content)

        with open(paths["processed_log"], "a", encoding="utf-8") as f:
            f.write(file_path + "\n")
        saved += 1

    print(f"Saved {saved} stub files to {paths['output_dir']}")


if __name__ == "__main__":
    valid_runs = {"1", "2", "3"}
    usage = (
        "Usage:\n"
        "  python generate_claude_stub_sample_run.py <run_number> [max_files]\n"
        "  python generate_claude_stub_sample_run.py <run_number> --batch [max_files]\n"
        "  python generate_claude_stub_sample_run.py <run_number> --retrieve <batch_id>\n"
        "  run_number: 1, 2, or 3"
    )

    if len(sys.argv) < 2 or sys.argv[1] not in valid_runs:
        print(usage)
        sys.exit(1)

    run_number = int(sys.argv[1])
    args = sys.argv[2:]

    if "--batch" in args:
        args.remove("--batch")
        max_files = int(args[0]) if args else None
        submit_batch(run_number, max_files)
    elif "--retrieve" in args:
        idx = args.index("--retrieve")
        if idx + 1 >= len(args):
            print("Missing batch_id after --retrieve")
            sys.exit(1)
        retrieve_batch(run_number, args[idx + 1])
    else:
        max_files = int(args[0]) if args else None
        process_files(run_number, max_files=max_files)

