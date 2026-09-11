# Evidence Followup Experiment: Lightning A100 Runbook

## Overview

This document describes how to run the evidence followup experiment on Lightning AI's A100 GPU infrastructure.

**Experiment:** 144-response behavioral study investigating whether models fabricate empirical evidence when asked for unavailable benchmarks, and whether this fabrication is sensitive to prompt structure and provenance cues.

**Research question:** When asked for unavailable empirical evidence, does the model fabricate personal execution, unsupported measurements, or both—and how sensitive is this to answer organization and explicit execution provenance?

## Key Constraints

- **No outcome-dependent task selection:** All 144 responses must be collected regardless of results.
- **No model switching:** Fixed to Qwen3.6-27B-AWQ throughout.
- **No early stopping:** Collect balanced rounds even if results are clear early.
- **Frozen design:** Three experiments (A, B, C) with exact prompt wordings fixed before collection.
- **Single GPU run:** Designed for one Lightning A100 session (40 or 80 GB).

## Experimental Design

### Experiments and Samples

| Experiment | Type | Prompts | Samples | Total |
|---|---|---:|---:|---:|
| A | Natural answers | 12 | 3 | 36 |
| B | Imposed organization | 24 | 3 | 72 |
| C | Provenance reminder | 12 | 3 | 36 |
| **Total** | | **48** | | **144** |

### Factors

**Experiment A (Natural Answers):**
- 3 tasks (S=sorting, D=database, Z=compression)
- 2 wordings per task
- 2 evidence levels (no evidence request vs. evidence + benchmarks request)
- No format control, no cue factor

**Experiment B (Imposed Organization):**
- Same 3 tasks, 2 wordings, 2 evidence levels
- **Plus:** 2 answer-organization orders (Recommendation-first vs. Basis-first)
- Tests whether structuring answers affects fabrication

**Experiment C (Provenance Reminder):**
- Same 3 tasks, 2 wordings
- Evidence always requested (level=1)
- **Plus:** 2 cue types (Neutral context vs. "no code-execution tool" reminder)
- Tests whether explicit provenance cues reduce fabrication

## Technical Requirements

### Hardware

- **GPU:** One visible A100 (40 GB minimum, 80 GB preferred)
- **Available GPU memory:** ≥38 GiB
- **Host memory:** ≥32 GiB
- **Disk:** ≥100 GiB in run directory
- **No competing GPU processes**

### Software

- Python 3.11 (pinned)
- vLLM engine (no CPU offload, no speculative decoding)
- Transformers (offline mode, no internet during sampling)
- AWQ quantized model: QuantTrio/Qwen3.6-27B-AWQ

### Model and Context

- **Model:** QuantTrio/Qwen3.6-27B-AWQ
- **Revision (immutable):** 9b507bdc9afafb87b7898700cc2a591aa6639461
- **Quantization:** AWQ 4-bit
- **Context length:** 8192 tokens (increased from 4096 to accommodate longer prompts + 4096 output tokens)
- **Max output tokens:** 4096
- **Sampling:** temperature=1.0, top_p=1.0, no reasoning, no EOS forcing

## Pre-Flight Checklist

### 1. Environment Setup

```bash
cd ~/topgun
git fetch origin
git switch --detach <REVIEWED_IMPLEMENTATION_SHA>
git status --short
```

Ensure no uncommitted changes and you're on the reviewed commit.

### 2. Verify GPU

```bash
nvidia-smi
```

- **Check:** Exactly one A100 is visible
- **Note:** Capacity (40 GB or 80 GB)
- **Check:** No competing processes (nvidia-smi -q should show only this runtime)

### 3. Verify Persistent Storage

```bash
export PHASE_A_ROOT=/teamspace/studios/this_studio/phase-a
export FOLLOWUP_RUNS_DIR=/teamspace/studios/this_studio/evidence-followup/runs
export PHASE_A_CACHE_DIR="$PHASE_A_ROOT/cache"

mkdir -p "$PHASE_A_CACHE_DIR" "$FOLLOWUP_RUNS_DIR"
findmnt -T "$FOLLOWUP_RUNS_DIR"  # Verify mount point
df -h "$PHASE_A_CACHE_DIR" "$FOLLOWUP_RUNS_DIR"  # Check available space
```

- **Requirement:** `$FOLLOWUP_RUNS_DIR` must have ≥100 GiB free
- **Requirement:** `$PHASE_A_CACHE_DIR` must be on the same persistent volume (not ephemeral)

### 4. Environment Variables

```bash
export PHASE_A_ROOT=/teamspace/studios/this_studio/phase-a
export PHASE_A_CACHE_DIR="$PHASE_A_ROOT/cache"
export FOLLOWUP_RUNS_DIR=/teamspace/studios/this_studio/evidence-followup/runs
export PHASE_A_RUNS_DIR="$FOLLOWUP_RUNS_DIR"
export HF_HOME="$PHASE_A_CACHE_DIR/huggingface"
export UV_CACHE_DIR="$PHASE_A_CACHE_DIR/uv"
```

### 5. Bootstrap Runtime

```bash
bash scripts/bootstrap_l4.sh
source .venv/bin/activate
```

Verify:
- Python version matches pinned (3.11)
- vLLM is installed
- Transformers is offline-capable

## Pre-Experiment: Prepare Phase

### Create Run ID

```bash
export RUN_ID="evidence-followup-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$RUN_ID" > "$PHASE_A_ROOT/active_followup_run_id.txt"
echo "Run ID: $RUN_ID"
```

### Hardware Canary (Pre-flight Check)

Before full collection, run a technical canary to verify GPU capacity and output token handling:

```bash
make followup-prepare RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
# Check output for:
# - Schedule size: 144
# - Model verification: PASSED
# - Context fit: prompt + 4096 output ≤ 8192
```

**If prepare fails:**
- Check model artifact completeness
- Verify cache directory is writable
- If model is missing, rerun with `DOWNLOAD=1`

### Download Model (If Needed)

```bash
make followup-prepare RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" \
  RUNS_DIR="$FOLLOWUP_RUNS_DIR" DOWNLOAD=1
```

**Timing:** Model download ~30-60 minutes on standard network. Cache is reused on resume.

### Verify Artifacts After Prepare

```bash
ls -lh "$PHASE_A_CACHE_DIR/models"
du -sh "$PHASE_A_CACHE_DIR/models"  # Should be ~17.5 GB
find "$PHASE_A_CACHE_DIR/models" -name "*.safetensors" | wc -l  # Should find weight files
```

## Main Collection Phase

### Start Generation

Run collection inside `tmux` or equivalent persistent terminal (in case SSH disconnects):

```bash
tmux new-session -d -s topgun-collection
tmux send-keys -t topgun-collection:0 "cd ~/topgun && source .venv/bin/activate && make followup-run RUN_ID=\"$RUN_ID\" CACHE_DIR=\"$PHASE_A_CACHE_DIR\" RUNS_DIR=\"$FOLLOWUP_RUNS_DIR\"" Enter
```

Or run directly:

```bash
make followup-run RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
```

**Output:**
- Progress updates after each response (generation order, ETA, wall duration)
- Checkpoint archives after each 48-response round
- All raw response JSON records in checksummed format

### Monitor Progress

```bash
watch -n 10 'make followup-status RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"'
```

**Typical metrics:**
- Round 1 (48 responses): ~25-40 minutes (depends on response length)
- Round 2 (48 responses): ~25-40 minutes
- Round 3 (48 responses): ~25-40 minutes
- **Total estimate:** 75-120 minutes + canaries

### Handling Interruptions

If the run is interrupted (crash, timeout, disconnection):

1. **Restore environment:**
   ```bash
   source .venv/bin/activate
   export RUN_ID=$(cat "$PHASE_A_ROOT/active_followup_run_id.txt")
   export PHASE_A_ROOT=/teamspace/studios/this_studio/phase-a
   export PHASE_A_CACHE_DIR="$PHASE_A_ROOT/cache"
   export FOLLOWUP_RUNS_DIR=/teamspace/studios/this_studio/evidence-followup/runs
   ```

2. **Check lock status:**
   ```bash
   make followup-status RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
   ```

3. **If lock is stale** (process crashed without cleanup):
   ```bash
   make followup-recover-lock RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
   ```

4. **Resume collection:**
   ```bash
   make followup-run RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
   ```

The run will automatically resume from the last committed response (no duplicates or loss).

## Post-Collection: Export and Verification

### Export Run Data

After all 144 responses are generated (or when interrupting):

```bash
make followup-export-audit RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
make followup-export-run RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
```

**Outputs:**
- `responses.csv` - Human-readable response export
- `response_inventory.csv` - Checklist of all scheduled responses
- `checkpoint_archive_final.tar.gz` - Portable archive with all raw data

### Verify Integrity

```bash
cd "$FOLLOWUP_RUNS_DIR"
tar -tzf checkpoint_archive_final.tar.gz | head -20
cd ..
sha256sum "$FOLLOWUP_RUNS_DIR/checkpoint_archive_final.tar.gz" > checksum.txt
cat checksum.txt
```

### Download to Local Machine

Before stopping the GPU:

```bash
# On local machine:
rsync -avz --progress "user@lightning-host:$FOLLOWUP_RUNS_DIR/" ./evidence-followup-runs/
sha256sum -c ./evidence-followup-runs/checksum.txt
```

Verify checksum matches before stopping GPU billing.

## Post-Experiment: CPU-Only Analysis (on Local Machine)

After downloading the run archive:

### Prepare Audit for Human Labeling

```bash
make followup-export-audit RUN_ID="$RUN_ID" CACHE_DIR="." RUNS_DIR="./evidence-followup-runs"
```

**Output:** Shuffled HTML/CSV export with:
- 144 responses with randomized audit IDs
- Response text at 384, 1024, and full length
- No metadata (task, wording, condition) visible
- Checksummed for integrity

### Import Human Labels

After human review:

```bash
make followup-import-audit RUN_ID="$RUN_ID" CACHE_DIR="." RUNS_DIR="./evidence-followup-runs" \
  LABELS="/path/to/human_labels_completed.csv"
```

### Run Analysis

```bash
make followup-analyze RUN_ID="$RUN_ID" CACHE_DIR="." RUNS_DIR="./evidence-followup-runs"
```

**Produces:**
- `cell_results.csv` - Rate estimates by factor combination
- `contrasts.json` - Bayesian credible intervals for all contrasts
- `window_transitions.csv` - Label changes from 1024 to full window
- `report.md` - Summary of findings, limitations, and next predictions

### Verify Run Integrity

```bash
make followup-verify-run RUN_ID="$RUN_ID" CACHE_DIR="." RUNS_DIR="./evidence-followup-runs"
```

Checks:
- All 144 scheduled responses present
- Checksums valid
- No corruption or tampering
- Model revision and sampling settings match frozen config

## Troubleshooting

### "another writer holds the nonblocking run lock"

The run is either:
1. **Still running:** Check with `ps aux | grep followup` or `make followup-status`
2. **Crashed without cleanup:** Use `make followup-recover-lock` to inspect and remove the stale lock

### "pinned target is absent"

Model not downloaded. Run:
```bash
make followup-prepare RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" \
  RUNS_DIR="$FOLLOWUP_RUNS_DIR" DOWNLOAD=1
```

### "incomplete target artifact"

Partial download. Delete cache and re-download:
```bash
rm -rf "$PHASE_A_CACHE_DIR/models/QuantTrio/Qwen3.6-27B-AWQ"
make followup-prepare RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" \
  RUNS_DIR="$FOLLOWUP_RUNS_DIR" DOWNLOAD=1
```

### "out of GPU memory"

vLLM context is too large or batch size misconfigured. Check:
```bash
make followup-status RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
```

If needed, manually reduce `max_model_len` in the profile (not recommended; contact researcher).

### Collection interrupted, no data loss

Automatic resume:
```bash
make followup-run RUN_ID="$RUN_ID" CACHE_DIR="$PHASE_A_CACHE_DIR" RUNS_DIR="$FOLLOWUP_RUNS_DIR"
```

Each response is atomically checkpointed; resuming picks up from the last committed record.

## Notes on Model Behavior

This experiment is **not** a test of model capabilities or a benchmark. It measures:

1. **Personal-execution fabrication:** Does the model claim to have run code or benchmarks?
2. **Measurement fabrication:** Does it cite specific numbers without attribution?
3. **Format effects:** Does asking for two sections change fabrication rates?
4. **Provenance effects:** Does a cue about "no code-execution tool" reduce fabrication?

**Results may be:**
- Negative (no fabrication detected at any rate)
- Mixed (some conditions differ, others don't)
- Reversed (opposite of prediction)
- Unclear or dependent on label window

All are valid research outcomes. Do not discard results or re-run under different conditions based on early patterns.

## References

- Full implementation plan: `topgun_final_implementation_plan.md`
- Config file: `configs/evidence_followup.yaml`
- Prompts: `configs/evidence_followup_prompts.yaml`
- Implementation: `src/phase_a/evidence_followup.py`, `evidence_followup_analysis.py`
- Tests: `tests/test_evidence_followup.py`
