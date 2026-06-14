"""
Submit 500 sample files (+ L2 dependency slices) to Claude Opus 4.6 via
Anthropic Message Batches API, then retrieve .pyi stubs.

Usage:
    python generate_claude_l2_stub_sample_run.py --submit           # all 500 files
    python generate_claude_l2_stub_sample_run.py --submit 5         # 5 largest (sanity check)
    python generate_claude_l2_stub_sample_run.py --retry-failed     # resubmit failed_files.txt only
    python generate_claude_l2_stub_sample_run.py --retrieve <batch_id>
    python generate_claude_l2_stub_sample_run.py --retrieve         # uses last batch_id.txt
"""

import json
import os
import sys

import anthropic
import tiktoken
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PARENT_DIR)

load_dotenv(os.path.join(REPO_ROOT, ".env"))

if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env in the repo root.")

client = anthropic.Anthropic()

MODEL_NAME = "claude-opus-4-6"
MAX_OUTPUT_TOKENS = 64000
RUN_NAME = "claude_opus_l2_stub_run"
OUTPUT_DIR = os.path.join(PARENT_DIR, RUN_NAME)
BATCH_ID_FILE = os.path.join(OUTPUT_DIR, "batch_id.txt")
FAILED_LOG = os.path.join(OUTPUT_DIR, "failed_files.txt")
MISSING_SLICE_LOG = os.path.join(OUTPUT_DIR, "missing_l2_slice.txt")

SELECTED_FILES_JSON = os.path.join(SCRIPT_DIR, "selected_500_files.json")
GROUPED_JSON = os.path.join(
    PARENT_DIR, "Files_not_for_root_directories", "grouped_file_paths.json"
)
L2_SLICE_DIR = os.path.join(PARENT_DIR, "dependency_slices", "L2")

SYSTEM_PROMPT = """You are generating a Python type stub (.pyi) file for static type checking.

You will also receive an L2 dependency slice: a read-only .pyi-style summary of imported
internal and third-party symbols (with types where available). Do NOT reproduce, modify,
or output the dependency slice.

Type resolution priority (use the first source that applies):
1. L2 dependency slice — for imported symbols (e.g. HomeAssistant, registry types, helpers).
2. Source code analysis — infer from usage, returns, defaults, docstrings, and patterns
   (e.g. str() → str, =None → Optional[...], =True → bool, iteration → Iterable, pytest
   fixtures by parameter name such as caplog, snapshot, entity_registry).
3. Standard library and well-known third-party APIs.
4. Any — ONLY as a last resort when there is genuinely no evidence for a more specific type.

Produce a .pyi stub for the given module that follows these rules:

1. Preserve all public API:
   - Include all top-level functions, classes, methods, and public variables that appear in the module.
   - Keep the same names and parameter structures as in the original code.
2. Use standard Python type hints (PEP 484) suitable for mypy and other type checkers.
3. Be as SPECIFIC as possible. Avoid Any for parameters and return types when a concrete type
   or a type from the L2 slice can be justified. Do not annotate everything as Any.
4. Use proper .pyi stub syntax:
   - Function and method bodies must be '...'.
   - Class bodies may contain method/attribute declarations with '...'.
   - Module-level variables should be annotated with a type and assigned '...'.
5. Include all necessary imports for the types you use in the stub.
6. Do NOT include any executable code, logic, or imports that are only used at runtime.
7. Do NOT add explanations, comments, or extra prose; output only stub code.
8. The output must be a single, complete .pyi file corresponding to this module.

Return only valid .pyi stub code."""


def _repo_rel_path(file_path: str) -> str:
    return file_path.replace("\\", "/")


def _source_path(file_path: str) -> str:
    return os.path.join(PARENT_DIR, _repo_rel_path(file_path))


def _selected_file_paths(largest_n: int | None = None) -> list[str]:
    with open(SELECTED_FILES_JSON, encoding="utf-8") as f:
        selected = set(json.load(f)["files"])
    with open(GROUPED_JSON, encoding="utf-8") as f:
        file_map = json.load(f)

    paths: list[str] = []
    for group_id in sorted(file_map.keys(), key=int):
        for file_path in file_map[group_id]:
            if os.path.basename(_repo_rel_path(file_path)) in selected:
                paths.append(file_path)

    if largest_n is None:
        return paths

    sized = [
        (os.path.getsize(_source_path(fp)), fp)
        for fp in paths
        if os.path.isfile(_source_path(fp))
    ]
    sized.sort(reverse=True)
    return [fp for _, fp in sized[:largest_n]]


def _custom_id(file_path: str) -> str:
    """Anthropic batch custom_id: ^[a-zA-Z0-9_-]{1,64}$ (not full repo paths)."""
    return os.path.splitext(os.path.basename(_repo_rel_path(file_path)))[0]


def _custom_id_to_path_map(file_paths: list[str] | None = None) -> dict[str, str]:
    if file_paths is None:
        file_paths = _selected_file_paths()
    return {_custom_id(fp): fp for fp in file_paths}


def _l2_slice_path(file_path: str) -> str:
    return os.path.join(L2_SLICE_DIR, f"{_custom_id(file_path)}.capsule.pyi")


def _token_count(text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _max_tokens(prompt: str, *, use_max_cap: bool = False) -> int:
    if use_max_cap:
        return MAX_OUTPUT_TOKENS
    prompt_tokens = _token_count(prompt)
    return min(MAX_OUTPUT_TOKENS, max(4096, prompt_tokens * 2))


def _user_message(code: str, dependency_slice: str, module_name: str) -> str:
    return (
        f"Here is the implementation of a Python module named {module_name!r}:\n\n"
        f"<<<PYTHON MODULE START>>>\n{code}\n<<<PYTHON MODULE END>>>\n\n"
        f"L2 dependency slice (read-only context — do not output this):\n\n"
        f"<<<L2 DEPENDENCY SLICE START>>>\n"
        f"{dependency_slice}\n"
        f"<<<L2 DEPENDENCY SLICE END>>>"
    )


def _read_inputs(file_path: str) -> tuple[str, str, str] | str:
    """Return (code, slice, module_name) or an error reason string."""
    slice_path = _l2_slice_path(file_path)
    if not os.path.isfile(slice_path):
        return "missing_l2_slice"

    source_path = _source_path(file_path)
    try:
        with open(source_path, encoding="utf-8", errors="ignore") as f:
            code = f.read()
        with open(slice_path, encoding="utf-8", errors="ignore") as f:
            dependency_slice = f.read()
    except OSError as e:
        return f"read_error: {e}"

    module_name = os.path.splitext(os.path.basename(_repo_rel_path(file_path)))[0]
    return code, dependency_slice, module_name


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def _write_lines(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _failed_file_paths() -> list[str]:
    if not os.path.isfile(FAILED_LOG):
        return []
    paths: list[str] = []
    with open(FAILED_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(line.split("\t", 1)[0])
    return paths


def submit_batch(
    file_paths: list[str] | None = None,
    *,
    largest_n: int | None = None,
    label: str | None = None,
    use_max_cap: bool = False,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if file_paths is None:
        file_paths = _selected_file_paths(largest_n)
    if label:
        cap_note = f", max_tokens={MAX_OUTPUT_TOKENS}" if use_max_cap else ""
        print(f"{label}: submitting {len(file_paths)} files{cap_note}")
    elif largest_n is not None:
        print(f"Sanity check: submitting {len(file_paths)} largest files:")
        for fp in file_paths:
            size_kb = os.path.getsize(_source_path(fp)) // 1024
            print(f"  {size_kb:>4} KB  {os.path.basename(_repo_rel_path(fp))}")

    requests: list[dict] = []
    missing_slice: list[str] = []
    failed: list[str] = []

    for file_path in file_paths:
        inputs = _read_inputs(file_path)
        if isinstance(inputs, str):
            if inputs == "missing_l2_slice":
                missing_slice.append(file_path)
            else:
                failed.append(f"{file_path}\t{inputs}")
            continue

        code, dependency_slice, module_name = inputs
        user_msg = _user_message(code, dependency_slice, module_name)
        prompt = SYSTEM_PROMPT + user_msg
        requests.append({
            "custom_id": _custom_id(file_path),
            "params": {
                "model": MODEL_NAME,
                "max_tokens": _max_tokens(prompt, use_max_cap=use_max_cap),
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

    if missing_slice:
        with open(MISSING_SLICE_LOG, "a", encoding="utf-8") as f:
            for line in missing_slice:
                f.write(line + "\n")
    for entry in failed:
        print(f"SKIP: {entry}")

    if not requests:
        print("No requests to submit.")
        return

    print(f"Submitting batch with {len(requests)} requests …")
    batch = client.messages.batches.create(requests=requests)

    with open(BATCH_ID_FILE, "w", encoding="utf-8") as f:
        f.write(batch.id)

    print(f"Submitted batch {batch.id} (status={batch.processing_status})")
    if missing_slice:
        print(f"Skipped {len(missing_slice)} files with no L2 slice (see {MISSING_SLICE_LOG})")
    print(f"Retrieve with:\n  python {sys.argv[0]} --retrieve {batch.id}")


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(getattr(block, "text", str(block)) for block in content)
    return str(content)


def retrieve_batch(batch_id: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch {batch_id}  status={batch.processing_status}")

    if batch.processing_status != "ended":
        print("Batch not finished yet. Try again later.")
        return

    saved = 0
    failed: list[str] = []

    id_to_path = _custom_id_to_path_map()
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        file_path = id_to_path.get(custom_id, custom_id)
        if result.result.type != "succeeded":
            error = getattr(result.result, "error", None)
            msg = getattr(error, "message", None) if error else None
            reason = msg or result.result.type
            print(f"FAILED: {file_path} — {reason}")
            failed.append(f"{file_path}\t{reason}")
            continue

        msg = result.result.message
        if msg.stop_reason == "max_tokens":
            print(f"FAILED: {file_path} — truncated (hit max_tokens)")
            failed.append(f"{file_path}\ttruncated (hit max_tokens)")
            continue

        stub = _strip_markdown_fences(_message_text(msg))
        if not stub.strip():
            print(f"FAILED: {file_path} — empty stub")
            failed.append(f"{file_path}\tempty stub")
            continue

        out_name = os.path.splitext(os.path.basename(_repo_rel_path(file_path)))[0] + ".pyi"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(stub)
        saved += 1

    _write_lines(FAILED_LOG, failed)
    print(f"Saved {saved} stubs to {OUTPUT_DIR}")
    if failed:
        print(f"{len(failed)} failures logged to {FAILED_LOG}")


def retry_failed() -> None:
    file_paths = _failed_file_paths()
    if not file_paths:
        print(f"No failed files in {FAILED_LOG}")
        return
    _write_lines(FAILED_LOG, [])
    submit_batch(file_paths, label="Retry failed", use_max_cap=True)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "--submit":
        largest_n = None
        if len(sys.argv) > 2:
            try:
                largest_n = int(sys.argv[2])
            except ValueError:
                print("Usage: python generate_claude_l2_stub_sample_run.py --submit [N]")
                sys.exit(1)
            if largest_n <= 0:
                print("N must be a positive integer.")
                sys.exit(1)
        submit_batch(largest_n=largest_n)
    elif command == "--retry-failed":
        retry_failed()
    elif command == "--retrieve":
        batch_id = sys.argv[2] if len(sys.argv) > 2 else None
        if not batch_id and os.path.isfile(BATCH_ID_FILE):
            with open(BATCH_ID_FILE, encoding="utf-8") as f:
                batch_id = f.read().strip()
        if not batch_id:
            print("Provide a batch_id or run --submit first.")
            sys.exit(1)
        retrieve_batch(batch_id)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
