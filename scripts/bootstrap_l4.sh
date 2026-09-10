#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Warning: running as root; an unprivileged user is preferred." >&2
fi

python3.11 -m venv .bootstrap-venv
. .bootstrap-venv/bin/activate
python -m pip install --disable-pip-version-check "uv==0.12.12"

export PHASE_A_CACHE_DIR="${PHASE_A_CACHE_DIR:-$PWD/.phase_a_cache}"
export PHASE_A_RUNS_DIR="${PHASE_A_RUNS_DIR:-$PWD/.phase_a_runs}"
export HF_HOME="${HF_HOME:-$PHASE_A_CACHE_DIR/huggingface}"
export UV_CACHE_DIR="$PHASE_A_CACHE_DIR/uv"

echo "Resolved cache directory: $PHASE_A_CACHE_DIR"
echo "Resolved runs directory: $PHASE_A_RUNS_DIR"
mkdir -p "$PHASE_A_CACHE_DIR" "$PHASE_A_RUNS_DIR"

uv sync --frozen --extra gpu --group dev
. .venv/bin/activate

python --version
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python -c "import vllm, transformers; print('vLLM', vllm.__version__); print('Transformers', transformers.__version__)"
nvidia-smi
python -c "from pathlib import Path; import os; from phase_a.environment import environment_report; environment_report(Path(os.environ['PHASE_A_CACHE_DIR']) / 'bootstrap_environment.json')"
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts

echo "Bootstrap complete. No model or dataset artifacts were downloaded."
