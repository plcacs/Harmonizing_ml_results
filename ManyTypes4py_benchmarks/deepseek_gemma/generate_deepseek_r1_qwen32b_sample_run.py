"""
Run DeepSeek-R1-Distill-Qwen-32B type annotation on the 500 selected sample files.

This script starts a local vLLM OpenAI-compatible server and then sends the same
requests as generate_gpt5_sample_run.py.

Usage:
    python generate_deepseek_r1_qwen32b_sample_run.py

    default run_number: 2
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import tiktoken
from openai import OpenAI
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
MODEL_SHORT_NAME = "deepseek_r1_qwen32b"
DEFAULT_CACHE_MODEL_DIR = (
    "/work/lzhan011/vllm/hf_cache/"
    "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"
)
DEFAULT_CACHE_WEIGHT_DIR = (
    "/work/lzhan011/vllm/hf_cache/hub/"
    "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"
)

VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8003"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", f"http://{VLLM_HOST}:{VLLM_PORT}/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
REQUEST_TIMEOUT = float(os.getenv("VLLM_REQUEST_TIMEOUT", "1200"))
SERVER_START_TIMEOUT = int(os.getenv("VLLM_SERVER_START_TIMEOUT", "1800"))
RUN_NUMBER = int(os.getenv("DEEPSEEK_R1_QWEN32B_RUN_NUMBER", "2"))

SELECTED_FILES_JSON = os.path.join(SCRIPT_DIR, "selected_500_files.json")
GROUPED_JSON = os.path.join(
    PARENT_DIR, "Files_not_for_root_directories", "grouped_file_paths.json"
)


def _repo_rel_path(file_path: str) -> str:
    """grouped_file_paths.json uses '\\' separators; normalize for POSIX basename/join."""
    return file_path.replace("\\", "/")


def _latest_snapshot(model_dir: Path) -> Path | None:
    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0]

    return None


def _link_snapshot_files(source_dir: Path, target_dir: Path) -> None:
    for source in source_dir.iterdir():
        if not source.is_file() and not source.is_symlink():
            continue
        target = target_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve())


def build_local_model_dir() -> str | None:
    config_snapshot = _latest_snapshot(Path(DEFAULT_CACHE_MODEL_DIR))
    weight_snapshot = _latest_snapshot(Path(DEFAULT_CACHE_WEIGHT_DIR))
    if not config_snapshot or not weight_snapshot:
        return None

    local_dir = Path(
        os.getenv(
            "DEEPSEEK_R1_QWEN32B_LOCAL_MODEL_DIR",
            f"/tmp/deepseek_r1_qwen32b_model_{os.getenv('SLURM_JOB_ID', 'local')}",
        )
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    _link_snapshot_files(config_snapshot, local_dir)
    _link_snapshot_files(weight_snapshot, local_dir)

    if (local_dir / "config.json").exists() and (local_dir / "model.safetensors.index.json").exists():
        return str(local_dir)
    return None


def resolve_model_path() -> str:
    configured_env = os.getenv("DEEPSEEK_R1_QWEN32B_MODEL_PATH")
    if configured_env:
        configured = Path(configured_env)
        if configured.exists():
            snapshot = _latest_snapshot(configured)
            return str(snapshot or configured)
        return configured_env

    local_model_dir = build_local_model_dir()
    if local_model_dir:
        return local_model_dir

    return MODEL_NAME


def wait_for_server() -> None:
    models_url = f"{VLLM_BASE_URL.rstrip('/')}/models"
    deadline = time.time() + SERVER_START_TIMEOUT
    last_error = ""

    while time.time() < deadline:
        try:
            request = urllib.request.Request(
                models_url,
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(5)

    raise RuntimeError(
        f"vLLM server did not become ready at {models_url} within "
        f"{SERVER_START_TIMEOUT}s. Last error: {last_error}"
    )


def start_vllm_server() -> subprocess.Popen[str] | None:
    if os.getenv("VLLM_SKIP_SERVER_START", "").lower() in {"1", "true", "yes"}:
        wait_for_server()
        return None

    model_path = resolve_model_path()
    os.makedirs(LOGS_DIR, exist_ok=True)
    server_log_path = os.path.join(LOGS_DIR, f"{MODEL_SHORT_NAME}_vllm_server.log")
    server_log = open(server_log_path, "a", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "2"),
        "--dtype",
        os.getenv("VLLM_DTYPE", "bfloat16"),
        "--gpu-memory-utilization",
        os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-model-len",
        os.getenv("VLLM_MAX_MODEL_LEN", "32768"),
        "--trust-remote-code",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    print(f"vLLM server log: {server_log_path}")
    process = subprocess.Popen(
        cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_for_server()
    return process


def get_client() -> OpenAI:
    return OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)


def get_run_paths(run_number: int) -> dict[str, str]:
    run_name = f"{MODEL_SHORT_NAME}_{run_number}_run"
    os.makedirs(LOGS_DIR, exist_ok=True)
    return {
        "output_dir": os.path.join(PARENT_DIR, run_name),
        "processed_log": os.path.join(LOGS_DIR, f"processed_files_{run_name}.txt"),
        "timing_log": os.path.join(LOGS_DIR, f"{run_name}_model_timings.json"),
        "unprocessed_log": os.path.join(LOGS_DIR, f"unprocessed_files_{run_name}.txt"),
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


def progress_iter(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, total=len(items), unit="file", dynamic_ncols=True)


def strip_model_markup(text: str) -> str:
    stripped = text.strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def generate_type_annotated_code(client: OpenAI, code: str) -> tuple[str, bool, float]:
    prompt = f"""Here is a Python program:

{code}

Add Python type annotations to the existing code.

Rules:
1. Only add type annotations to function parameters and return types.
2. Do not modify the program logic or existing code.
3. Do not add explanations, comments, or extra text.
4. Output only the complete annotated Python program.

Return only valid Python code.
Use standard Python type hints (PEP 484)."""
    max_retries = 3
    wait_time = 60
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a python programming expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=REQUEST_TIMEOUT,
            )
            duration = time.time() - start_time
            content = response.choices[0].message.content or ""
            return strip_model_markup(content), True, duration
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            if "tokens per min" in error_msg:
                print("TPM limit hit - not retrying.")
                return code, False, 0
            if "rate_limit_exceeded" in error_msg:
                print(
                    f"Rate limit. Retrying in {wait_time}s "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                wait_time += 30
                continue
            return code, False, 0
    print("Max retries reached. Skipping.")
    return code, False, 0


def process_files(run_number: int) -> None:
    server_process = start_vllm_server()
    client = get_client()
    paths = get_run_paths(run_number)

    try:
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

        total = len(files_to_run)
        already_done = sum(1 for _, fp in files_to_run if fp in processed_files)
        remaining = total - already_done
        print(
            f"DeepSeek-R1-Qwen32B run {run_number}: {total} selected files, "
            f"{already_done} already done, {remaining} remaining"
        )

        pending_files = [(group_id, fp) for group_id, fp in files_to_run if fp not in processed_files]

        processed_count = 0
        for group_id, file_path in progress_iter(
            pending_files, f"DeepSeek run {run_number}"
        ):
            print(f"Processing: {file_path}")
            full_path = os.path.join(PARENT_DIR, _repo_rel_path(file_path))
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except (UnicodeDecodeError, IOError) as e:
                print(f"Read error, skipping: {e}")
                continue

            content, success, duration = generate_type_annotated_code(client, code)
            log_timing(file_path, duration, paths["timing_log"])

            if not success or not content.strip():
                print(f"Skipping {file_path} - generation failed")
                with open(paths["unprocessed_log"], "a", encoding="utf-8") as f:
                    f.write(file_path + "\n")
                continue

            out_dir = os.path.join(paths["output_dir"], group_id)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, os.path.basename(_repo_rel_path(file_path)))
            try:
                with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(content)
            except (UnicodeEncodeError, IOError) as e:
                print(f"Write error, skipping: {e}")
                continue

            with open(paths["processed_log"], "a", encoding="utf-8") as f:
                f.write(file_path + "\n")

            processed_count += 1
            print(f"Done [{processed_count}/{remaining}]: {file_path}")
            time.sleep(5)
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    valid_runs = {"2", "3", "4"}
    if str(RUN_NUMBER) not in valid_runs:
        print("Usage: python generate_deepseek_r1_qwen32b_sample_run.py")
        print("  run_number: 2, 3, or 4")
        print(
            "  configure with DEEPSEEK_R1_QWEN32B_RUN_NUMBER "
            "(default: 2)"
        )
        sys.exit(1)

    process_files(RUN_NUMBER)
