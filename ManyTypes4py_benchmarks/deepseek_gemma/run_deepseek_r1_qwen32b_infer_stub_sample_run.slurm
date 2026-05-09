#!/bin/bash
#SBATCH --job-name=mt4py_ds32b_stub
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --account=loni_irt4lln01
#SBATCH --output=/work/lzhan011/vllm/Harmonizing_ml_results/ManyTypes4py_benchmarks/deepseek_gemma/logs/%x-%j.out
#SBATCH --error=/work/lzhan011/vllm/Harmonizing_ml_results/ManyTypes4py_benchmarks/deepseek_gemma/logs/%x-%j.err

set -euo pipefail

WORKDIR="/work/lzhan011/vllm"
CACHEDIR="${WORKDIR}/hf_cache"
RUN_LOGDIR="${WORKDIR}/Harmonizing_ml_results/ManyTypes4py_benchmarks/deepseek_gemma/logs"
mkdir -p "${RUN_LOGDIR}"

module purge
module load cuda/11.8.0
export CUDA_HOME=/usr/local/packages/cuda/11.8.0/7oedjh4
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

VENV="/work/lzhan011/conda_envs/vllm"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "ERROR: ${VENV}/bin/python not found or not executable"
  exit 1
fi
export PATH="${VENV}/bin:${PATH}"

export HF_HOME="${CACHEDIR}"
export TRANSFORMERS_CACHE="${CACHEDIR}"
export HUGGINGFACE_HUB_CACHE="${CACHEDIR}/hub"
export TOKENIZERS_PARALLELISM=false
if [ -f "${CACHEDIR}/token" ]; then
  export HF_TOKEN="$(cat "${CACHEDIR}/token")"
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

unset DEEPSEEK_R1_QWEN32B_MODEL_PATH
export DEEPSEEK_R1_QWEN32B_LOCAL_MODEL_DIR="${DEEPSEEK_R1_QWEN32B_LOCAL_MODEL_DIR:-${SLURM_TMPDIR:-/tmp}/deepseek_r1_qwen32b_model_${SLURM_JOB_ID}}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8003}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_REQUEST_TIMEOUT="${VLLM_REQUEST_TIMEOUT:-1200}"
export VLLM_SERVER_START_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-1800}"

export DEEPSEEK_R1_QWEN32B_INFER_STUB_RUN_NUMBER="${DEEPSEEK_R1_QWEN32B_INFER_STUB_RUN_NUMBER:-1}"

echo "Local model path: ${DEEPSEEK_R1_QWEN32B_LOCAL_MODEL_DIR}"
echo "Endpoint: ${VLLM_BASE_URL}"
if [ -n "${DEEPSEEK_R1_QWEN32B_INFER_STUB_MAX_FILES:-}" ]; then
  echo "Max files: ${DEEPSEEK_R1_QWEN32B_INFER_STUB_MAX_FILES}"
fi

cd "${WORKDIR}/Harmonizing_ml_results/ManyTypes4py_benchmarks/deepseek_gemma"

MODEL_PATH="$(python - <<'PY'
from pathlib import Path
import os

cache_dir = Path("/work/lzhan011/vllm/hf_cache")
config_root = cache_dir / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"
weight_root = cache_dir / "hub" / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"


def latest_snapshot(root):
    snapshots_dir = root / "snapshots"
    snapshots = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0]


def link_files(source_dir, target_dir):
    for source in source_dir.iterdir():
        if not source.is_file() and not source.is_symlink():
            continue
        target = target_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve())


local_dir = Path(os.environ["DEEPSEEK_R1_QWEN32B_LOCAL_MODEL_DIR"])
local_dir.mkdir(parents=True, exist_ok=True)
link_files(latest_snapshot(config_root), local_dir)
link_files(latest_snapshot(weight_root), local_dir)

if not (local_dir / "config.json").exists():
    raise SystemExit(f"missing config.json in {local_dir}")
if not (local_dir / "model.safetensors.index.json").exists():
    raise SystemExit(f"missing model.safetensors.index.json in {local_dir}")

print(local_dir)
PY
)"

SERVER_LOG="${RUN_LOGDIR}/deepseek_r1_qwen32b_infer_stub_${SLURM_JOB_ID}_server.log"
echo "Resolved model path: ${MODEL_PATH}"
echo "Server log: ${SERVER_LOG}"

"${VENV}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --dtype "${VLLM_DTYPE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --trust-remote-code \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

python - <<'PY'
import os
import time
import urllib.request

url = os.environ["VLLM_BASE_URL"].rstrip("/") + "/models"
deadline = time.time() + int(os.environ.get("VLLM_SERVER_START_TIMEOUT", "1800"))
last_error = ""
while time.time() < deadline:
    try:
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                print(f"vLLM server is ready: {url}", flush=True)
                raise SystemExit
    except Exception as exc:
        last_error = str(exc)
    time.sleep(5)
raise SystemExit(f"vLLM server did not become ready: {last_error}")
PY

RUN_NUMBERS="${DEEPSEEK_R1_QWEN32B_INFER_STUB_RUN_NUMBERS:-1 2 3}"
for RUN_NUMBER in ${RUN_NUMBERS}; do
  export VLLM_SKIP_SERVER_START=1
  export DEEPSEEK_R1_QWEN32B_INFER_STUB_RUN_NUMBER="${RUN_NUMBER}"
  echo "Run number: ${DEEPSEEK_R1_QWEN32B_INFER_STUB_RUN_NUMBER}"
  python generate_deepseek_r1_qwen32b_infer_stub_sample_run.py
done
