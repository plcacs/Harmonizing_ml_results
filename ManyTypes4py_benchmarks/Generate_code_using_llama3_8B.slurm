#!/bin/bash
#SBATCH --job-name=mt4py_l3_8b
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --account=loni_irt4lln01
#SBATCH --output=/work/lzhan011/vllm/logs/%x-%j.out
#SBATCH --error=/work/lzhan011/vllm/logs/%x-%j.err

set -euo pipefail

WORKDIR="/work/lzhan011/vllm"
CACHEDIR="${WORKDIR}/hf_cache"

module purge
module load cuda/11.8.0
export CUDA_HOME=/usr/local/packages/cuda/11.8.0/7oedjh4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

VENV="/work/lzhan011/conda_envs/vllm"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "ERROR: ${VENV}/bin/python not found or not executable"
  exit 1
fi
export PATH="${VENV}/bin:${PATH}"

export HF_HOME="${CACHEDIR}"
export TRANSFORMERS_CACHE="${CACHEDIR}"
export HUGGINGFACE_HUB_CACHE="${CACHEDIR}"
export TOKENIZERS_PARALLELISM=false

# Endpoint config: default to current Llama-3 8B vLLM service.
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://qbd514:8002/v1}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

echo "Using endpoint: ${VLLM_BASE_URL}"
echo "Using model: ${VLLM_MODEL_NAME}"

cd "${WORKDIR}/Harmonizing_ml_results/ManyTypes4py_benchmarks"
python Generate_code_using_llama3_8B.py
