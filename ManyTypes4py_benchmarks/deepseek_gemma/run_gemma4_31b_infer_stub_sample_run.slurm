#!/bin/bash
#SBATCH --job-name=mt4py_gemma31b_stub
#SBATCH --partition=gpu2
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
module load cuda/12.2.1
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CC="/work/lzhan011/conda_envs/gcc11_real/bin/x86_64-conda-linux-gnu-gcc"
export CXX="/work/lzhan011/conda_envs/gcc11_real/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"
export CMAKE_C_COMPILER="${CC}"
export CMAKE_CXX_COMPILER="${CXX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

VENV="/work/lzhan011/conda_envs/vllm_gemma"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "ERROR: ${VENV}/bin/python not found or not executable"
  exit 1
fi
export PATH="${VENV}/bin:${PATH}"

export HF_HOME="${CACHEDIR}"
export TRANSFORMERS_CACHE="${CACHEDIR}"
export HUGGINGFACE_HUB_CACHE="${CACHEDIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -f "${CACHEDIR}/token" ]; then
  export HF_TOKEN="$(cat "${CACHEDIR}/token")"
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

export GEMMA4_31B_MODEL_PATH="${GEMMA4_31B_MODEL_PATH:-${CACHEDIR}/hub/models--google--gemma-4-31B-it}"
export GEMMA4_31B_SERVER_DIR="${GEMMA4_31B_SERVER_DIR:-${WORKDIR}/Harmonizing_ml_results/ManyTypes4py_benchmarks/deepseek_gemma}"
export PYTHONPATH="${GEMMA4_31B_SERVER_DIR}:${PYTHONPATH:-}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8004}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_REQUEST_TIMEOUT="${VLLM_REQUEST_TIMEOUT:-1200}"
export VLLM_SERVER_START_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-1800}"
export GEMMA4_31B_MAX_TOKENS="${GEMMA4_31B_MAX_TOKENS:-4096}"
export GEMMA4_31B_DEVICE_MAP="${GEMMA4_31B_DEVICE_MAP:-auto}"

export GEMMA4_31B_INFER_STUB_RUN_NUMBER="${GEMMA4_31B_INFER_STUB_RUN_NUMBER:-1}"

echo "Model path: ${GEMMA4_31B_MODEL_PATH}"
echo "Gemma server dir: ${GEMMA4_31B_SERVER_DIR}"
echo "Endpoint: ${VLLM_BASE_URL}"
if [ -n "${GEMMA4_31B_INFER_STUB_MAX_FILES:-}" ]; then
  echo "Max files: ${GEMMA4_31B_INFER_STUB_MAX_FILES}"
fi

cd "${WORKDIR}/Harmonizing_ml_results/ManyTypes4py_benchmarks/deepseek_gemma"

MODEL_PATH="$(python - <<'PY'
from pathlib import Path
import os

configured = Path(os.environ["GEMMA4_31B_MODEL_PATH"])
snapshots_dir = configured / "snapshots"
if snapshots_dir.is_dir():
    snapshots = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if snapshots:
        print(snapshots[0])
        raise SystemExit
print(configured)
PY
)"

SERVER_LOG="${RUN_LOGDIR}/gemma4_31b_infer_stub_${SLURM_JOB_ID}_server.log"
echo "Resolved model path: ${MODEL_PATH}"
echo "Server log: ${SERVER_LOG}"

export MODEL_ID="google/gemma-4-31B-it"
"${VENV}/bin/python" -m uvicorn gemma4_31B_transformers_server:app \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
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
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print(f"vLLM server is ready: {url}", flush=True)
                raise SystemExit
    except Exception as exc:
        last_error = str(exc)
    time.sleep(5)
raise SystemExit(f"vLLM server did not become ready: {last_error}")
PY

for RUN_NUMBER in 1 2 3; do
  export VLLM_SKIP_SERVER_START=1
  export GEMMA4_31B_INFER_STUB_RUN_NUMBER="${RUN_NUMBER}"
  echo "Run number: ${GEMMA4_31B_INFER_STUB_RUN_NUMBER}"
  python generate_gemma4_31b_infer_stub_sample_run.py
done
