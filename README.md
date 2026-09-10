# Phase A: WeirdChat reproduction

This repository implements the candidate-selection phase described in [phase_a_implementation_plan.md](phase_a_implementation_plan.md). It selects fixed, relatively benign WeirdChat cases and measures whether they reproduce under one pinned local Qwen3.6-27B configuration. It does not minimize or transfer triggers, and it never uses hosted inference.

## Scientific and cost boundaries

- Target and judge inference run locally through vLLM on one NVIDIA L4.
- WeirdChat v1.0.1, target models, tokenizers, and the judge are pinned to immutable Hugging Face revisions.
- Production generation defaults to Hugging Face offline mode and aborts when a remote backend, non-loopback URL, or common hosted-provider credential is selected.
- FP8 is recorded as an `exact_configuration_attempt`; the third-party 4-bit AWQ fallback is recorded as `quantized_reproduction`. Their runs must never be pooled.
- Generated data, downloaded artifacts, caches, credentials, and bundles are ignored by Git.

Pinned WeirdChat v1.0.1 has at most eight distinct behaviors after the required safety exclusions, so the reserve list may be short; the pipeline records that shortfall rather than weakening safety or the one-pattern-per-behavior rule.

## Development verification

Use Python 3.11 or newer:

```bash
python -m pip install uv==0.12.12
uv sync --frozen --group dev
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

The CPU suite uses tiny fixtures and mock inference. It does not download models or require a GPU.

Before a production run, commit and push reviewed code, record `git rev-parse HEAD`, clone that exact commit on the L4, and confirm `git status --short` is empty. The run refuses a dirty tree; `--allow-dirty` is for development only and must not be used for production.

## L4 setup and artifact lock

```bash
git clone https://github.com/ArhaanShah/topgun.git
cd topgun
git checkout <GREEN_CI_COMMIT_SHA>
test -z "$(git status --short)"

nvidia-smi
free -h
df -h .
command -v python3.11

export PHASE_A_ROOT="<VERIFIED_PERSISTENT_DIRECTORY>/phase-a"
export PHASE_A_CACHE_DIR="$PHASE_A_ROOT/cache"
export PHASE_A_RUNS_DIR="$PHASE_A_ROOT/runs"
export HF_HOME="$PHASE_A_CACHE_DIR/huggingface"
mkdir -p "$PHASE_A_CACHE_DIR" "$PHASE_A_RUNS_DIR"

./scripts/bootstrap_l4.sh
make preflight PROFILE=fp8_offload
make download PROFILE=fp8_offload
make verify-offline PROFILE=fp8_offload

export RUN_ID="phase-a-$(date -u +%Y%m%dT%H%M%SZ)"
make prepare PROFILE=fp8_offload RUN_ID="$RUN_ID"
make health-check PROFILE=fp8_offload RUN_ID="$RUN_ID"
make canary PROFILE=fp8_offload RUN_ID="$RUN_ID"
```

Inspect the canary text, timing, VRAM, and RAM report before continuing. If FP8 health checking fails, use `PROFILE=awq_4bit` with a new run ID and start from zero. Never mix profiles.

## Production workflow

Run the remaining gated commands in their exact required order.

```bash
make smoke PROFILE=fp8_offload RUN_ID="$RUN_ID"
make judge-smoke PROFILE=fp8_offload RUN_ID="$RUN_ID"
make export-audit RUN_ID="$RUN_ID"
```

Download the standalone audit directory and complete `human_audit.csv` without consulting automated labels or published rates. Then, import the labels to unlock reproduction:

```bash
python -m phase_a.audit import-labels --run "$PHASE_A_RUNS_DIR/$RUN_ID" --labels human_audit_completed.csv
make reproduce PROFILE=fp8_offload RUN_ID="$RUN_ID"
make judge-reproduction PROFILE=fp8_offload RUN_ID="$RUN_ID"
make finalize RUN_ID="$RUN_ID"
make export-run RUN_ID="$RUN_ID"
python scripts/verify_bundle.py "$PHASE_A_RUNS_DIR/$RUN_ID/phase_a_${RUN_ID}.tar.zst"
```

If reliability fails, finalization writes `audit/manual_label_required.csv`; label every reproduction response for that candidate and import the completed superset before finalizing again. Human labels override automated labels. The final selector takes the first four qualifying candidates in the committed deterministic order, or reports `INSUFFICIENT_REPRODUCIBLE_CASES` without relaxing gates.

Also note:
- Public Hugging Face downloads and local inference require no paid inference API.
- Lightning compute/storage may still consume the operator's credits.
- `HF_TOKEN` is optional and must never be written to a manifest or bundle.
- Raw responses and model caches must not be committed to GitHub.
- Before shutting down an ephemeral instance, export the allowlisted run bundle and checksum, download both, and verify the checksum on another machine.

## Runtime layout

Caches default to `.phase_a_cache/`; run state defaults to `.phase_a_runs/<run_id>/`. A run contains configuration and environment manifests, selections, rendered prompts, responses, judgments, a blinded audit package, reports, and crash-recovery checkpoints. Export is allowlisted and excludes model weights, datasets, caches, credentials, and logs.
