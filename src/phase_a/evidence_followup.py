"""Evidence followup experiment: frozen 144-response design with three experiments.

GPU imports are deliberately confined to production prepare/run paths so audit and
analysis remain usable on an ordinary CPU installation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import random
import tarfile
import tempfile
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import hash_small_files
from .config import REPO_ROOT, artifact_path, load_profile, load_yaml
from .environment import enable_offline_mode, git_info
from .inference import Backend, Completion, make_backend, render_user_prompt
from .storage import atomic_write_json, atomic_write_text, canonical_json, compute_checksum

CONFIG_PATH = REPO_ROOT / "configs" / "evidence_followup.yaml"
PROMPTS_PATH = REPO_ROOT / "configs" / "evidence_followup_prompts.yaml"
PROFILE_NAME = "qwen3_6_27b_awq_a100_followup"
PROFILE_PATH = REPO_ROOT / "configs" / "models" / f"{PROFILE_NAME}.yaml"
LOCK_PATH = "manifests/writer.lock"

# Experimental design constants
TASK_IDS = ["S", "D", "Z"]
EXPERIMENTS = {"A": "natural", "B": "imposed_organization", "C": "provenance_reminder"}
TRUE_VALUES = {"1", "true", "yes", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "n", "f"}


def _label_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES or text == "":
        return False if text in FALSE_VALUES else None
    raise ValueError(f"invalid Boolean label {value!r}")


class SimpleTokenizer:
    """Small tokenizer implementing the production interface for CPU mocks."""

    name_or_path = "evidence-followup-mock-tokenizer"
    chat_template = "<|user|>{{ content }}<|assistant|>"

    def __len__(self) -> int:
        return 65536

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(
        self,
        token_ids: list[int] | list[str],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(str(item) for item in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        if len(messages) != 1 or messages[0]["role"] != "user" or kwargs.get("enable_thinking") is not False:
            raise ValueError("followup prompts must be one user turn with thinking disabled")
        return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_value(value: Any) -> str:
    return compute_checksum(canonical_json(value))


def derive_experiment_seed(master_seed: int, namespace: str, *parts: object) -> int:
    """Derive a deterministic seed from master seed and factors."""
    payload = "||".join((str(master_seed), namespace, *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & 0x7FFFFFFF


def response_identity(
    experiment_id: str,
    experiment: str,
    task: str,
    wording: int,
    evidence: int,
    order: str | None,
    cue: str | None,
    replicate: int,
) -> str:
    """Compute unique response identity."""
    payload = (
        f"{experiment_id}||{experiment}||{task}||{wording}||{evidence}||{order or 'none'}||{cue or 'none'}||{replicate}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def load_experiment_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and validate frozen design configuration."""
    config = load_yaml(CONFIG_PATH)
    prompts = load_yaml(PROMPTS_PATH)
    profile = load_profile(PROFILE_NAME)

    if config.get("profile") != PROFILE_NAME:
        raise ValueError(f"experiment profile must be {PROFILE_NAME}")
    if config.get("prediction_note", {}).get("frozen") is not True:
        raise ValueError("prediction_note.frozen must be true")
    if config.get("sampling") != {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096, "reasoning": False}:
        raise ValueError("sampling settings differ from the frozen followup design")
    if profile.get("max_model_len") != 8192 or profile.get("generation_config") != "vllm":
        raise ValueError("A100 followup profile must use 8192 context and vLLM generation defaults")

    # Validate design totals
    if config.get("design", {}).get("total_responses") != 144:
        raise ValueError("the frozen design must contain exactly 144 responses")
    if config.get("design", {}).get("total_unique_prompts") != 48:
        raise ValueError("the frozen design must contain exactly 48 unique prompts")

    return config, prompts, profile


def render_prompt(
    config: dict[str, Any],
    prompts: dict[str, Any],
    experiment: str,
    task: str,
    wording: int,
    evidence: int = 0,
    order: str | None = None,
    cue: str | None = None,
) -> str:
    """Render a complete prompt from task template and factors."""
    separator = "\n\n"

    # Start with base task
    task_key = f"tasks.{task}.wording_{wording}"
    base_text = prompts.get("tasks", {}).get(task, {}).get(f"wording_{wording}", "")
    if not base_text:
        raise ValueError(f"Task prompt not found: {task_key}")

    parts = [base_text]

    # Add evidence request
    parts.append(prompts.get("evidence_common", ""))
    if evidence == 1:
        parts.append(prompts.get("evidence_benchmark", ""))

    # Add format instructions (Experiment B only)
    if experiment == "B":
        if order == "recommendation-first":
            parts.append(prompts.get("format_recommendation_first", ""))
        elif order == "basis-first":
            parts.append(prompts.get("format_basis_first", ""))
        else:
            raise ValueError(f"Invalid order for experiment B: {order}")

    # Add cue instructions (Experiment C only)
    if experiment == "C":
        if cue == "neutral":
            parts.append(prompts.get("cue_neutral", ""))
        elif cue == "execution-unavailable":
            parts.append(prompts.get("cue_execution_unavailable", ""))
        else:
            raise ValueError(f"Invalid cue for experiment C: {cue}")

    return separator.join(part.strip() for part in parts if part.strip())


def build_schedule(config: dict[str, Any], prompts: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the full 144-response schedule with three balanced rounds."""
    schedule: list[dict[str, Any]] = []
    seeds: set[int] = set()
    ids: set[str] = set()

    # Define all 48 unique prompt variants
    variants = []

    # Experiment A: Natural answers (12 prompts)
    for task in TASK_IDS:
        for wording in [0, 1]:
            for evidence in [0, 1]:
                variants.append(
                    {
                        "experiment": "A",
                        "task": task,
                        "wording": wording,
                        "evidence": evidence,
                        "order": None,
                        "cue": None,
                    }
                )

    # Experiment B: Imposed organization (24 prompts)
    for task in TASK_IDS:
        for wording in [0, 1]:
            for evidence in [0, 1]:
                for order in ["recommendation-first", "basis-first"]:
                    variants.append(
                        {
                            "experiment": "B",
                            "task": task,
                            "wording": wording,
                            "evidence": evidence,
                            "order": order,
                            "cue": None,
                        }
                    )

    # Experiment C: Provenance reminder (12 prompts)
    for task in TASK_IDS:
        for wording in [0, 1]:
            for cue in ["neutral", "execution-unavailable"]:
                variants.append(
                    {
                        "experiment": "C",
                        "task": task,
                        "wording": wording,
                        "evidence": 1,  # C always has evidence=1
                        "order": None,
                        "cue": cue,
                    }
                )

    if len(variants) != 48:
        raise AssertionError(f"the frozen design must have 48 unique prompts, got {len(variants)}")

    # Create 3 balanced rounds (144 responses total, 48 per round)
    for round_num in range(3):
        block = list(range(len(variants)))
        random.Random(derive_experiment_seed(config["master_seed"], "order", round_num)).shuffle(block)

        for within_block, variant_idx in enumerate(block):
            variant = variants[variant_idx]
            factors = (
                variant["experiment"],
                variant["task"],
                variant["wording"],
                variant["evidence"],
                variant.get("order"),
                variant.get("cue"),
                round_num,
            )

            seed = derive_experiment_seed(config["master_seed"], "sample", *factors)
            response_id = response_identity(config["experiment_id"], *factors)

            if seed in seeds or response_id in ids:
                raise AssertionError("response seeds and IDs must be unique")
            seeds.add(seed)
            ids.add(response_id)

            schedule.append(
                {
                    "response_id": response_id,
                    "experiment": variant["experiment"],
                    "task": variant["task"],
                    "wording": variant["wording"],
                    "evidence": variant["evidence"],
                    "order": variant.get("order"),
                    "cue": variant.get("cue"),
                    "replicate": round_num,
                    "block": round_num,
                    "within_block": within_block,
                    "generation_order": len(schedule),
                    "seed": seed,
                    "raw_prompt": render_prompt(
                        config,
                        prompts,
                        variant["experiment"],
                        variant["task"],
                        variant["wording"],
                        variant["evidence"],
                        variant.get("order"),
                        variant.get("cue"),
                    ),
                }
            )

    if len(schedule) != 144:
        raise AssertionError(f"the frozen schedule must contain exactly 144 responses, got {len(schedule)}")

    # Verify each round contains all 48 variants
    for round_num in range(3):
        round_variants = [
            (row["experiment"], row["task"], row["wording"], row["evidence"], row["order"], row["cue"])
            for row in schedule
            if row["block"] == round_num
        ]
        if len(set(round_variants)) != 48:
            raise AssertionError(f"round {round_num} must cover all 48 variants")

    return schedule


def _require_explicit_paths(
    run_id: str | None, cache_dir: str | Path | None, runs_dir: str | Path | None
) -> tuple[Path, Path]:
    """Validate and expand explicit paths for the run."""
    if not run_id or not str(run_id).strip():
        raise ValueError("--run-id is required and must be nonempty")
    if cache_dir is None or not str(cache_dir).strip() or runs_dir is None or not str(runs_dir).strip():
        raise ValueError("--cache-dir and --runs-dir are required and must be nonempty")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be one safe path component")
    return Path(cache_dir).expanduser().resolve(), Path(runs_dir).expanduser().resolve() / run_id


class ProcessLock(AbstractContextManager["ProcessLock"]):
    """Nonblocking exclusive write lock for run directory."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / LOCK_PATH
        self.acquired = False

    def __enter__(self) -> ProcessLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"another writer holds the nonblocking run lock: {self.path}; "
                "use the recover-lock command only after a hard shutdown"
            ) from exc
        import psutil

        process = psutil.Process(os.getpid())
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "process_create_time": process.create_time(),
            "host": platform.node(),
            "boot_id": _boot_id(),
            "created_at": _utc_now(),
        }
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, *args: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def _boot_id() -> str:
    """Get a machine-persistent boot identifier."""
    linux_boot_id = Path("/proc/sys/kernel/random/boot_id")
    if linux_boot_id.is_file():
        return linux_boot_id.read_text(encoding="utf-8").strip()
    import psutil

    return f"{platform.node()}:{psutil.boot_time():.6f}"


def _read_lock_payload(data: bytes, path: Path) -> dict[str, Any]:
    """Parse lock file content, with backward compatibility."""
    try:
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("pid"), int):
            return payload
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    # Backward compatibility for legacy lock format
    try:
        fields = dict(part.split("=", 1) for part in data.decode("utf-8").split() if "=" in part)
        return {"pid": int(fields["pid"]), "legacy": True}
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot safely recover malformed writer lock: {path}") from exc


def recover_stale_lock(run_dir: Path) -> dict[str, Any]:
    """Remove a lock only after proving its recorded process is no longer its owner."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run does not exist: {run_dir}")
    path = run_dir / LOCK_PATH
    if not path.exists():
        return {"status": "NO_LOCK", "path": str(path)}
    original = path.read_bytes()
    payload = _read_lock_payload(original, path)
    pid = payload["pid"]
    if pid <= 0:
        raise RuntimeError(f"cannot safely recover writer lock with invalid PID: {path}")
    current_boot = _boot_id()
    recorded_boot = payload.get("boot_id")
    reason: str | None = None
    if recorded_boot and recorded_boot != current_boot:
        reason = "recorded operating-system boot has ended"
    else:
        import psutil

        try:
            process = psutil.Process(pid)
            recorded_start = payload.get("process_create_time")
            if recorded_start is not None and abs(process.create_time() - float(recorded_start)) > 0.01:
                reason = "PID was reused by a different process"
            elif process.status() == psutil.STATUS_ZOMBIE:
                reason = "recorded process is a zombie and cannot write"
            else:
                raise RuntimeError(f"refusing to remove writer lock: recorded PID {pid} is still the owning process")
        except psutil.NoSuchProcess:
            reason = "recorded PID no longer exists"
        except psutil.AccessDenied as exc:
            raise RuntimeError(f"cannot verify whether recorded PID {pid} is gone; lock retained") from exc
    if path.read_bytes() != original:
        raise RuntimeError("writer lock changed during recovery; lock retained")
    path.unlink()
    return {"status": "STALE_LOCK_REMOVED", "path": str(path), "pid": pid, "verification": reason}


def _model_provenance(local: Path, profile: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    """Check model artifact provenance and revision."""
    revision = profile["revision"]
    manifest_path = cache_dir / "artifact_manifest.json"
    sources: list[str] = []
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(
                item.get("repository") == profile["model"] and item.get("revision") == revision
                for item in old.get("artifacts", [])
            ):
                sources.append(str(manifest_path.resolve()))
        except (OSError, json.JSONDecodeError):
            pass
    for metadata in local.rglob("*.metadata"):
        try:
            if revision in metadata.read_text(encoding="utf-8", errors="ignore"):
                sources.append(str(metadata.relative_to(local)))
                break
        except OSError:
            continue
    explicit = local / ".evidence_followup_provenance.json"
    if explicit.exists():
        try:
            recorded = json.loads(explicit.read_text(encoding="utf-8"))
            if recorded.get("repository") == profile["model"] and recorded.get("revision") == revision:
                sources.append(str(explicit.relative_to(local)))
        except (OSError, json.JSONDecodeError):
            pass
    if not sources:
        raise RuntimeError(
            "the target directory has no recorded revision provenance; rerun prepare with --download "
            "(or make followup-prepare ... DOWNLOAD=1)"
        )
    return {"revision": revision, "provenance_sources": sources}


def verify_target(cache_dir: Path, profile: dict[str, Any], *, load_transformers: bool = True) -> dict[str, Any]:
    """Verify model artifact completeness and integrity."""
    local = artifact_path(cache_dir, profile["model"], profile["revision"])
    if not local.is_dir():
        raise RuntimeError(f"pinned target is absent: {local}; rerun prepare with --download")
    provenance = _model_provenance(local, profile, cache_dir)
    config_path = local / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"incomplete target artifact: missing {config_path.name}")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    quant = model_config.get("quantization_config", {})
    method = str(quant.get("quant_method", quant.get("method", ""))).lower()
    if method != "awq":
        raise RuntimeError(f"target quantization metadata is not AWQ: {method or 'missing'}")
    indexes = sorted(local.glob("*.index.json"))
    referenced: set[str] = set()
    for index in indexes:
        data = json.loads(index.read_text(encoding="utf-8"))
        referenced.update(data.get("weight_map", {}).values())
    if referenced:
        missing = sorted(name for name in referenced if not (local / name).is_file())
        if missing:
            raise RuntimeError(f"incomplete target artifact; missing {len(missing)} indexed weight shards")
    elif not list(local.glob("*.safetensors")):
        raise RuntimeError("incomplete target artifact; no indexed or standalone safetensors weights")
    if load_transformers:
        enable_offline_mode()
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    hashes, total_size = hash_small_files(local)
    metadata_hashes = {
        name: digest for name, digest in hashes.items() if (local / name).stat().st_size <= 16 * 1024 * 1024
    }
    metadata_files = {
        name: {"size_bytes": (local / name).stat().st_size, "sha256": digest}
        for name, digest in metadata_hashes.items()
    }
    return {
        "repository": profile["model"],
        "revision": profile["revision"],
        "path": str(local.resolve()),
        "resolved_size_bytes": total_size,
        "small_metadata_sha256": metadata_hashes,
        "small_metadata_files": metadata_files,
        "weight_shards_referenced": len(referenced),
        "integrity_scope": "presence/size plus hashes of small metadata; not full weight-file integrity",
        **provenance,
    }


def _download_target(cache_dir: Path, profile: dict[str, Any]) -> None:
    """Download model artifact if not already present."""
    from huggingface_hub import snapshot_download

    local = artifact_path(cache_dir, profile["model"], profile["revision"])
    snapshot_download(
        repo_id=profile["model"],
        revision=profile["revision"],
        local_dir=local,
        token=os.environ.get("HF_TOKEN"),
    )
    # A run-specific provenance marker
    atomic_write_json(
        local / ".evidence_followup_provenance.json",
        {"repository": profile["model"], "revision": profile["revision"], "downloaded_at": _utc_now()},
    )


def _load_tokenizer(cache_dir: Path, profile: dict[str, Any]) -> Any:
    """Load tokenizer from cached artifact."""
    enable_offline_mode()
    from transformers import AutoTokenizer

    local = artifact_path(cache_dir, profile["tokenizer"], profile["tokenizer_revision"])
    return AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)


def _snapshot_files(run_dir: Path) -> dict[str, str]:
    """Snapshot configuration files for reproducibility."""
    destination = run_dir / "config_snapshot"
    paths = (CONFIG_PATH, PROMPTS_PATH, PROFILE_PATH, REPO_ROOT / "uv.lock")
    hashes = {}
    for source in paths:
        text = source.read_text(encoding="utf-8")
        atomic_write_text(destination / source.name, text)
        hashes[source.name] = compute_checksum(text)
    return hashes


def _frozen_core(config: dict[str, Any], prompts: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Core frozen components for validation and recovery."""
    return {"config": config, "prompts": prompts, "profile": profile, "sampling": config["sampling"]}


def prepare_experiment(
    run_id: str,
    cache_dir: Path,
    runs_dir: Path,
    *,
    download: bool = False,
    mock: bool = False,
) -> dict[str, Any]:
    """Prepare a run: validate configs, build schedule, verify artifacts, set up directories."""
    run_dir = runs_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with ProcessLock(run_dir):
        return _prepare_experiment_locked(run_id, cache_dir, run_dir, download=download, mock=mock)


def _prepare_experiment_locked(
    run_id: str, cache_dir: Path, run_dir: Path, *, download: bool, mock: bool
) -> dict[str, Any]:

    # Load and validate config
    config, prompts, profile = load_experiment_config()

    # Build the frozen schedule
    schedule = build_schedule(config, prompts)

    # Record basic metadata
    git = git_info(REPO_ROOT)
    git_commit = git["commit_sha"]
    git_remote_url = git["remote_url"]
    git_dirty = bool(git["dirty"])
    if git_dirty and not mock:
        raise RuntimeError("production preparation requires a clean Git worktree")
    existing = run_dir / "prepare_manifest.json"
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("schedule_hash") == _hash_value(schedule) and bool(previous.get("mock")) == mock:
            return previous
        if any(run_dir.glob("responses/*.json")):
            raise RuntimeError("cannot change a prepared run after responses exist")
        raise RuntimeError("existing prepared run is incompatible; choose a new run ID")

    metadata: dict[str, Any] = {
        "status": "PREPARED",
        "run_id": run_id,
        "git_commit": git_commit,
        "git_remote": git_remote_url,
        "git_dirty": git_dirty,
        "schedule_size": len(schedule),
        "config_hashes": _snapshot_files(run_dir),
        "experiment_mode": "mock" if mock else "evidence-followup",
        "mock": mock,
        "profile": PROFILE_NAME,
        "sampling": config["sampling"],
        "schedule_hash": _hash_value(schedule),
    }

    if not mock:
        if download:
            try:
                _download_target(cache_dir, profile)
            except Exception as e:
                metadata["download_error"] = str(e)
                raise
        # Download must precede verification on a fresh machine.
        target_info = verify_target(cache_dir, profile, load_transformers=False)
        metadata["model_verification"] = target_info

    # Save immutable schedule rows and the rendered prompt/token accounting.
    schedule_path = run_dir / "schedule.jsonl"
    rendered_rows = []
    tokenizer = SimpleTokenizer()
    for row in schedule:
        atomic_write_json(schedule_path.parent / f"schedule_{row['generation_order']:04d}.json", row)
        rendered = render_user_prompt(tokenizer, row["raw_prompt"], profile["tokenizer_revision"])
        if rendered.rendered_token_count + config["sampling"]["max_tokens"] > profile["max_model_len"]:
            raise RuntimeError(f"prompt {row['response_id']} does not fit the frozen context")
        rendered_rows.append(
            {
                **row,
                "prompt": {
                    "rendered_prompt": rendered.rendered,
                    "raw_token_count": rendered.raw_token_count,
                    "rendered_token_count": rendered.rendered_token_count,
                    "rendered_hash": rendered.rendered_hash,
                    "tokenizer_hash": rendered.tokenizer_hash,
                },
            }
        )
    atomic_write_text(run_dir / "schedule.jsonl", "".join(canonical_json(row) + "\n" for row in rendered_rows))

    # Save manifest
    atomic_write_json(run_dir / "prepare_manifest.json", metadata)
    return metadata


def _schedule(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "schedule.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(path_.read_text(encoding="utf-8")) for path_ in sorted(run_dir.glob("schedule_*.json"))]


def _records(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "responses").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rid = record.get("response_id")
        if (
            not rid
            or rid in result
            or record.get("record_hash") != _hash_value({k: v for k, v in record.items() if k != "record_hash"})
        ):
            raise RuntimeError(f"invalid or duplicate response record: {path}")
        result[rid] = record
    return result


def _mock_completion(row: dict[str, Any]) -> Completion:
    count = 4096 if row["generation_order"] % 48 == 0 else (1100 if row["generation_order"] % 17 == 0 else 12)
    text = " ".join(f"token{i}" for i in range(count))
    return Completion(
        text, "length" if count == 4096 else "stop", row["prompt"]["rendered_token_count"], count, list(range(count))
    )


def run_experiment(
    run_dir: Path,
    cache_dir: Path,
    *,
    resume: bool,
    mock: bool = False,
    backend: Backend | None = None,
    tokenizer: Any | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest_path = run_dir / "prepare_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("prepare must complete before run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("mock")) != mock:
        raise RuntimeError("mock/production mode must match the prepared run")
    schedule = _schedule(run_dir)
    if len(schedule) != 144:
        raise RuntimeError("frozen schedule is incomplete")
    records = _records(run_dir)
    if records and not resume:
        raise RuntimeError("records already exist; pass --resume to continue")
    pending = [row for row in schedule if row["response_id"] not in records][:limit]
    if not pending:
        return status_run(run_dir)
    with ProcessLock(run_dir):
        tokenizer = tokenizer or (SimpleTokenizer() if mock else _load_tokenizer(cache_dir, load_profile(PROFILE_NAME)))
        if backend is None and not mock:
            profile = load_profile(PROFILE_NAME)
            runtime = dict(profile)
            runtime["model"] = str(artifact_path(cache_dir, profile["model"], profile["revision"]))
            runtime["tokenizer"] = str(artifact_path(cache_dir, profile["tokenizer"], profile["tokenizer_revision"]))
            backend = make_backend(runtime)
        for row in pending:
            started = time.monotonic()
            rendered = row.get("prompt", {}).get("rendered_prompt") or row["raw_prompt"]
            completion = (
                _mock_completion(row) if mock else backend.generate([rendered], [row["seed"]], manifest["sampling"])[0]
            )  # type: ignore[union-attr]
            payload = {
                "response_id": row["response_id"],
                "experiment": row["experiment"],
                "task": row["task"],
                "wording": row["wording"],
                "evidence": row["evidence"],
                "order": row["order"],
                "cue": row["cue"],
                "replicate": row["replicate"],
                "seed": row["seed"],
                "raw_prompt": row["raw_prompt"],
                "rendered_prompt": rendered,
                "prompt_token_count": completion.prompt_tokens,
                "output_token_ids": list(completion.token_ids or []),
                "response_text": completion.text,
                "decoded_384": " ".join(completion.text.split()[:384]),
                "decoded_1024": " ".join(completion.text.split()[:1024]),
                "completion_token_count": completion.completion_tokens,
                "finish_reason": completion.finish_reason,
                "duration_seconds": time.monotonic() - started,
                "generated_at": _utc_now(),
                "immutable_content_hash": compute_checksum(completion.text),
            }
            atomic_write_json(
                run_dir / "responses" / f"{row['response_id']}.json", {**payload, "record_hash": _hash_value(payload)}
            )
            records[row["response_id"]] = {**payload, "record_hash": _hash_value(payload)}
            if len(records) % 48 == 0:
                atomic_write_json(
                    run_dir / "checkpoints" / f"checkpoint_{len(records):03d}.json",
                    {"count": len(records), "response_ids": sorted(records), "created_at": _utc_now()},
                )
        atomic_write_text(
            run_dir / "responses.jsonl",
            "".join(
                canonical_json(records[row["response_id"]]) + "\n" for row in schedule if row["response_id"] in records
            ),
        )
    return status_run(run_dir)


def status_run(run_dir: Path) -> dict[str, Any]:
    schedule, records = _schedule(run_dir), _records(run_dir)
    failed_ids = {path.stem for path in (run_dir / "errors").glob("*.json")} if (run_dir / "errors").exists() else set()
    counts = {"planned": len(schedule), "successful": 0, "capped": 0, "missing": 0, "failed": 0, "pending": 0}
    for row in schedule:
        record = records.get(row["response_id"])
        if not record and row["response_id"] in failed_ids:
            counts["failed"] += 1
        elif not record:
            counts["pending"] += 1
        elif record.get("finish_reason") == "length" or record.get("completion_token_count") >= 4096:
            counts["capped"] += 1
        else:
            counts["successful"] += 1
    return {"status": "COMPLETE" if counts["pending"] == 0 else "IN_PROGRESS", **counts}


def export_run(run_dir: Path, output_dir: Path | None = None) -> Path:
    """Create a portable checksum-protected archive without model or environment files."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "prepare_manifest.json").exists():
        raise RuntimeError("prepared run is required")
    output_dir = Path(output_dir or run_dir.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"evidence_followup_{run_dir.name}.tar.gz"
    allowed = {
        "config_snapshot",
        "manifests",
        "responses",
        "audit",
        "reports",
        "checkpoints",
        "errors",
        "schedule.jsonl",
        "prepare_manifest.json",
        "responses.jsonl",
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.name in {"writer.lock"}:
                continue
            relative = path.relative_to(run_dir)
            if relative.parts[0] not in allowed:
                continue
            archive.add(path, arcname=f"{run_dir.name}/{relative.as_posix()}", recursive=False)
    digest = compute_checksum(archive_path.read_bytes())
    atomic_write_text(archive_path.with_name(archive_path.name + ".sha256"), f"{digest}  {archive_path.name}\n")
    return archive_path


def verify_run(archive_path: Path, extracted_run: Path | None = None) -> dict[str, Any]:
    """Verify checksum and extract into a caller-selected directory, then validate records."""
    archive_path = Path(archive_path).resolve()
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").split()[0] != compute_checksum(
        archive_path.read_bytes()
    ):
        raise ValueError("archive SHA-256 mismatch or missing sidecar")
    destination = Path(extracted_run).resolve() if extracted_run else Path(tempfile.mkdtemp(prefix="followup-verify-"))
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            pure = Path(member.name)
            if pure.is_absolute() or ".." in pure.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("archive must contain one run directory")
    run = roots[0]
    records = _records(run)
    schedule = _schedule(run)
    if any(rid not in {row["response_id"] for row in schedule} for rid in records):
        raise ValueError("archive contains a response outside the frozen schedule")
    return {
        "status": "VERIFIED",
        "run": str(run),
        "planned": len(schedule),
        "records": len(records),
        "complete": len(records) == len(schedule),
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    from .evidence_followup_analysis import ExperimentAnalyzer

    active = run_dir / "audit" / "active_labels.json"
    if not active.exists():
        raise RuntimeError("validated imported labels are required before analysis")
    labels = run_dir / "audit" / json.loads(active.read_text(encoding="utf-8"))["path"]
    responses = run_dir / "responses.jsonl"
    return ExperimentAnalyzer(responses, labels).analyze(run_dir / "reports")


def mock_e2e(run_id: str, cache_dir: Path, runs_dir: Path) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    prepare_experiment(run_id, cache_dir, run_dir, mock=True)
    run_experiment(run_dir, cache_dir, resume=True, mock=True)
    review = export_audit(run_dir)
    labels = run_dir / "audit" / "mock_labels.csv"
    with review.open(newline="", encoding="utf-8") as source, labels.open("w", newline="", encoding="utf-8") as target:
        rows = list(csv.DictReader(source))
        fields = list(rows[0])
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for field in fields:
                if field.startswith(("execution_claim_", "unsupported_measurements_", "ambiguous_", "retracted_")):
                    row[field] = "false"
                elif field.startswith("evidence_quote_"):
                    row[field] = ""
            writer.writerow(row)
    import_audit(run_dir, labels)
    report = analyze_run(run_dir)
    archive = export_run(run_dir, runs_dir)
    verified = verify_run(archive, runs_dir / f"{run_id}_verified")
    return {
        "status": "COMPLETE_MOCK",
        "run_dir": str(run_dir),
        "analysis": report,
        "archive": str(archive),
        "verification": verified,
    }


def export_audit(run_dir: Path) -> Path:
    records = _records(run_dir)
    schedule = _schedule(run_dir)
    target = run_dir / "audit" / "human_review.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "response_id",
        "immutable_content_hash",
        "saved_1024",
        "full_response",
        *(
            f"{field}_{window}"
            for field in ("execution_claim", "unsupported_measurements", "ambiguous", "retracted", "evidence_quote")
            for window in ("384", "1024", "full")
        ),
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in schedule:
            record = records.get(row["response_id"])
            if not record:
                continue
            writer.writerow(
                {
                    "response_id": row["response_id"],
                    "immutable_content_hash": record["immutable_content_hash"],
                    "saved_1024": record["decoded_1024"],
                    "full_response": record["response_text"],
                }
            )
    html_path = target.with_suffix(".html")
    html_path.write_text(
        "<html><body><p>Condition metadata and automated labels are intentionally hidden; "
        "responses may nevertheless reveal their condition.</p><table>"
        + "".join(
            "<tr><td>"
            + html.escape(row["response_id"])
            + "</td><td><pre>"
            + html.escape(row["saved_1024"])
            + "</pre></td></tr>"
            for row in csv.DictReader(target.open(encoding="utf-8"))
        )
        + "</table></body></html>",
        encoding="utf-8",
    )
    return target


def import_audit(run_dir: Path, labels_path: Path) -> Path:
    records = _records(run_dir)
    expected = {row["response_id"] for row in _schedule(run_dir)}
    with Path(labels_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("response_id", "") for row in rows]
    if set(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError("labels must contain every expected response ID exactly once")
    allowed = TRUE_VALUES | FALSE_VALUES | {""}
    for row in rows:
        rid = row["response_id"]
        if rid not in records:
            raise ValueError(f"{rid}: response has not been collected")
        if row.get("immutable_content_hash") != records[rid]["immutable_content_hash"]:
            raise ValueError(f"{rid}: immutable content hash mismatch")
        for key, value in row.items():
            if (
                key.startswith(("execution_claim_", "unsupported_measurements_", "ambiguous_", "retracted_"))
                and value.lower() not in allowed
            ):
                raise ValueError(f"{rid}: invalid label {key}")
            if (
                key.startswith("evidence_quote_")
                and _label_bool(row.get(f"execution_claim_{key.rsplit('_', 1)[-1]}")) is True
            ):
                if not value.strip() or value not in records[rid]["response_text"]:
                    raise ValueError(f"{rid}: positive labels require an exact evidence quote")
    audit = run_dir / "audit"
    audit.mkdir(exist_ok=True)
    version = len(list(audit.glob("imported_labels_v*.csv"))) + 1
    target = audit / f"imported_labels_v{version}.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(
        audit / "active_labels.json",
        {"path": target.name, "version": version, "sha256": compute_checksum(target.read_bytes())},
    )
    return target


# CLI commands follow the understand_2x2 pattern
def main() -> None:
    """Main CLI entry point for evidence followup experiment."""
    parser = argparse.ArgumentParser(description="Evidence followup experiment: 144-response design")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Prepare
    prep = subparsers.add_parser("prepare", help="Prepare experiment: validate configs and build schedule")
    prep.add_argument("--run-id", required=True, help="Unique run identifier")
    prep.add_argument("--cache-dir", required=True, help="Cache directory for models")
    prep.add_argument("--runs-dir", required=True, help="Runs base directory")
    prep.add_argument("--download", action="store_true", help="Download model if missing")
    prep.add_argument("--mock", action="store_true", help="Mock mode for testing")

    # Status
    status_cmd = subparsers.add_parser("status", help="Show run status")
    status_cmd.add_argument("--run-id", required=True, help="Run identifier")
    status_cmd.add_argument("--cache-dir", required=True, help="Cache directory")
    status_cmd.add_argument("--runs-dir", required=True, help="Runs directory")

    # Run
    run_cmd = subparsers.add_parser("run", help="Execute generation")
    run_cmd.add_argument("--run-id", required=True, help="Run identifier")
    run_cmd.add_argument("--cache-dir", required=True, help="Cache directory")
    run_cmd.add_argument("--runs-dir", required=True, help="Runs directory")
    run_cmd.add_argument("--mock", action="store_true", help="Mock mode")
    run_cmd.add_argument("--limit", type=int)
    run_cmd.add_argument("--resume", action="store_true")
    for name in ("export-audit", "analyze", "export-run", "verify-run", "recover-lock", "mock-e2e"):
        item = subparsers.add_parser(name)
        item.add_argument("--run-id", required=False)
        item.add_argument("--cache-dir", required=False)
        item.add_argument("--runs-dir", required=False)
        item.add_argument("--run", type=Path)
        if name == "export-run":
            item.add_argument("--output-dir", type=Path)
        if name == "verify-run":
            item.add_argument("--archive", type=Path)
            item.add_argument("--extract-to", type=Path)
    imp = subparsers.add_parser("import-audit")
    imp.add_argument("--run-id", required=True)
    imp.add_argument("--cache-dir", required=True)
    imp.add_argument("--runs-dir", required=True)
    imp.add_argument("--labels", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "prepare":
        cache_dir, runs_dir = _require_explicit_paths(args.run_id, args.cache_dir, args.runs_dir)
        result = prepare_experiment(
            args.run_id,
            cache_dir,
            runs_dir,
            download=args.download,
            mock=args.mock,
        )
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "status":
        cache_dir, runs_dir = _require_explicit_paths(args.run_id, args.cache_dir, args.runs_dir)
        print(json.dumps(status_run(runs_dir), indent=2))
    elif args.command == "run":
        _, runs_dir = _require_explicit_paths(args.run_id, args.cache_dir, args.runs_dir)
        print(
            json.dumps(
                run_experiment(
                    runs_dir,
                    Path(args.cache_dir).resolve(),
                    resume=args.resume if hasattr(args, "resume") else True,
                    mock=args.mock,
                    limit=args.limit,
                ),
                indent=2,
            )
        )
    elif args.command == "mock-e2e":
        _, runs_dir = _require_explicit_paths(args.run_id, args.cache_dir, args.runs_dir)
        print(json.dumps(mock_e2e(args.run_id, Path(args.cache_dir).resolve(), runs_dir.parent), indent=2))
    elif args.command in {"export-audit", "analyze", "export-run", "recover-lock", "import-audit"}:
        _, run_dir = _require_explicit_paths(args.run_id, args.cache_dir, args.runs_dir)
        if args.command == "export-audit":
            print(export_audit(run_dir))
        elif args.command == "import-audit":
            print(import_audit(run_dir, args.labels))
        elif args.command == "analyze":
            print(json.dumps(analyze_run(run_dir), indent=2))
        elif args.command == "recover-lock":
            print(json.dumps(recover_stale_lock(run_dir), indent=2))
        else:
            print(export_run(run_dir, args.output_dir))
    elif args.command == "verify-run":
        if not args.archive:
            raise ValueError("--archive is required")
        print(json.dumps(verify_run(args.archive, args.extract_to), indent=2))


if __name__ == "__main__":
    main()
