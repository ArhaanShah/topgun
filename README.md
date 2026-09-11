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

CPU setup can also be performed directly with `uv sync --frozen --python 3.11 --group dev`; omit the GPU extra on machines without CUDA.

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

## A100 2x2 understanding experiment

The separate `phase_a.understand_2x2` command implements the frozen 80-completion
experiment in `2x2_implementation_plan.md`. It uses only the pinned AWQ target—no
judge or WeirdChat download—and writes atomic source records outside the checkout.
CPU checks establish pipeline readiness only; the A100 preflight, engine load, and
1,024-token technical canary remain required hardware validation.

Select an A100 40 GB Lightning Studio first. On a new instance clone the repository;
in an existing Studio retain its checkout and model cache. Use a reviewed detached
commit and do not pull while a run is active:

```bash
cd ~/topgun
git fetch origin
git switch --detach origin/main
git rev-parse HEAD

export PHASE_A_ROOT="$HOME/phase-a"
export PHASE_A_CACHE_DIR="$PHASE_A_ROOT/cache"
export HF_HOME="$PHASE_A_CACHE_DIR/huggingface"
export UV_CACHE_DIR="$PHASE_A_CACHE_DIR/uv"
export TWO_BY_TWO_RUNS_DIR="$HOME/understand-2x2/runs"
mkdir -p "$PHASE_A_CACHE_DIR" "$TWO_BY_TWO_RUNS_DIR"

bash scripts/bootstrap_l4.sh
source .venv/bin/activate
nvidia-smi
df -h "$PHASE_A_CACHE_DIR" "$TWO_BY_TWO_RUNS_DIR"
```

Confirm where `$HOME` is mounted in the current Studio; a new Studio does not
automatically contain an earlier cache. Create and save one run ID outside the repo.
The committed prediction note is already frozen and `prepare` prints it for review:

```bash
export RUN_ID="understand-2x2-a100-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$HOME/understand-2x2"
printf '%s\n' "$RUN_ID" > "$HOME/understand-2x2/active_run_id.txt"

make 2x2-prepare RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR"
# Only when the pinned target is absent or incomplete:
make 2x2-prepare RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR" DOWNLOAD=1

make 2x2-run RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR"
make 2x2-export-audit RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR"
```

Use a persistent terminal such as tmux. Generation logs and checkpoints after every
response. After reconnecting, restore the environment and reuse exactly the saved ID:

```bash
export RUN_ID="$(cat "$HOME/understand-2x2/active_run_id.txt")"
make 2x2-run RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR"
```

If Lightning was hard-stopped while writing and the next invocation reports
`writer.lock`, first ensure no generation process from that run is active, then use:

```bash
make 2x2-recover-lock RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR"
```

This command does not offer a force option. It reads the recorded PID, process start
time, and operating-system boot identity and removes the lock only after proving that
the original process is gone (including a Studio reboot or PID reuse). It refuses to
delete an active, malformed, unverifiable, or concurrently changed lock. After a
successful recovery, rerun `make 2x2-run` with the same arguments and `RUN_ID`.

After all 80 outcomes, download the entire run folder as a backup and label from its
`audit/` folder. GPU compute may stop at this point. Upload the completed CSV without
changing its immutable columns, then import and analyze on any CPU machine holding a
copy of the run:

```bash
make 2x2-import-audit RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR" \
  LABELS="$HOME/topgun/human_labels_completed.csv"
make 2x2-analyze RUN_ID="$RUN_ID" \
  CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$TWO_BY_TWO_RUNS_DIR"
```

Label export and analysis never inspect the cache or GPU. Back up the updated whole
run folder afterward; an ordinary folder download or `tar.gz` is sufficient.
