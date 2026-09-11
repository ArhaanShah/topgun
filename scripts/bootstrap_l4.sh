#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Warning: running as root; an unprivileged user is preferred." >&2
fi

export PHASE_A_CACHE_DIR="${PHASE_A_CACHE_DIR:-$PWD/.phase_a_cache}"
export PHASE_A_RUNS_DIR="${PHASE_A_RUNS_DIR:-$PWD/.phase_a_runs}"
export HF_HOME="${HF_HOME:-$PHASE_A_CACHE_DIR/huggingface}"
export UV_CACHE_DIR="$PHASE_A_CACHE_DIR/uv"

echo "Resolved cache directory: $PHASE_A_CACHE_DIR"
echo "Resolved runs directory: $PHASE_A_RUNS_DIR"
mkdir -p "$PHASE_A_CACHE_DIR" "$PHASE_A_RUNS_DIR"

uv_is_pinned() {
  [[ -x "$1" ]] && [[ "$("$1" --version 2>/dev/null)" == "uv 0.12.12" ]]
}

if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
  if uv_is_pinned .venv/bin/uv; then
    UV_BIN="$PWD/.venv/bin/uv"
  elif command -v uv >/dev/null 2>&1 && uv_is_pinned "$(command -v uv)"; then
    UV_BIN="$(command -v uv)"
  else
    .venv/bin/python -m pip install --disable-pip-version-check "uv==0.12.12"
    UV_BIN="$PWD/.venv/bin/uv"
  fi
elif command -v uv >/dev/null 2>&1 && uv_is_pinned "$(command -v uv)"; then
  UV_BIN="$(command -v uv)"
else
  BOOTSTRAP_PYTHON="$(command -v python3 || command -v python || true)"
  if [[ -z "$BOOTSTRAP_PYTHON" ]]; then
    echo "No python3/python is available to bootstrap pinned uv." >&2
    exit 1
  fi
  "$BOOTSTRAP_PYTHON" -m venv .bootstrap-venv
  . .bootstrap-venv/bin/activate
  python -m pip install --disable-pip-version-check "uv==0.12.12"
  UV_BIN="$PWD/.bootstrap-venv/bin/uv"
fi

"$UV_BIN" python install 3.11
"$UV_BIN" sync --frozen --python 3.11 --extra gpu --group dev
. .venv/bin/activate

python --version
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python -c "import vllm, transformers; print('vLLM', vllm.__version__); print('Transformers', transformers.__version__)"
nvidia-smi
python -c "from pathlib import Path; import os; from phase_a.environment import environment_report; environment_report(Path(os.environ['PHASE_A_CACHE_DIR']) / 'bootstrap_environment.json')"

echo "Bootstrap complete. No model or dataset artifacts were downloaded."
