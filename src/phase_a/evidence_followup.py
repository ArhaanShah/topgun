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
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import hash_small_files
from .config import REPO_ROOT, artifact_path, load_profile, load_yaml
from .environment import enable_offline_mode, environment_report, git_info, package_versions, path_is_writable
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


def _snapshot_source_hashes() -> dict[str, str]:
    return {
        source.name: compute_checksum(source.read_text(encoding="utf-8"))
        for source in (CONFIG_PATH, PROMPTS_PATH, PROFILE_PATH, REPO_ROOT / "uv.lock")
    }


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

    metadata: dict[str, Any] = {
        "status": "PREPARED",
        "run_id": run_id,
        "git_commit": git_commit,
        "git_remote": git_remote_url,
        "git_dirty": git_dirty,
        "schedule_size": len(schedule),
        "config_hashes": _snapshot_source_hashes(),
        "experiment_mode": "mock" if mock else "evidence-followup",
        "mock": mock,
        "profile": PROFILE_NAME,
        "sampling": config["sampling"],
        "raw_schedule_hash": _hash_value(schedule),
        "frozen_manifest_hash": _hash_value(_frozen_core(config, prompts, profile)),
        "lockfile_hash": compute_checksum((REPO_ROOT / "uv.lock").read_bytes()),
        "engine": profile,
        "model": profile["model"],
        "model_revision": profile["revision"],
        "tokenizer": profile["tokenizer"],
        "tokenizer_revision": profile["tokenizer_revision"],
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
    rendered_rows = []
    tokenizer = SimpleTokenizer() if mock else _load_tokenizer(cache_dir, profile)
    for row in schedule:
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
                    "raw_hash": rendered.raw_hash,
                    "rendered_hash": rendered.rendered_hash,
                    "tokenizer_hash": rendered.tokenizer_hash,
                    "chat_template_hash": rendered.chat_template_hash,
                },
            }
        )
    metadata["rendered_schedule_hash"] = _hash_value(rendered_rows)
    metadata["tokenizer_hash"] = rendered_rows[0]["prompt"]["tokenizer_hash"]
    metadata["chat_template_hash"] = rendered_rows[0]["prompt"]["chat_template_hash"]
    audit_order = [row["response_id"] for row in rendered_rows]
    random.Random(derive_experiment_seed(config["master_seed"], "audit-order")).shuffle(audit_order)
    metadata["audit_order_hash"] = _hash_value(audit_order)
    if not mock and getattr(tokenizer, "name_or_path", "") == SimpleTokenizer.name_or_path:
        raise RuntimeError("production preparation refuses mock tokenizer provenance")
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in metadata.items() if k != "config_hashes"}
        prior = {k: previous.get(k) for k in comparable}
        snapshots_ok = all(
            (run_dir / "config_snapshot" / name).is_file()
            and compute_checksum((run_dir / "config_snapshot" / name).read_bytes()) == digest
            for name, digest in previous.get("config_hashes", {}).items()
        )
        existing_order = run_dir / "audit" / "review_order.json"
        if (prior == comparable and snapshots_ok and _schedule(run_dir) == rendered_rows
                and existing_order.is_file()
                and json.loads(existing_order.read_text(encoding="utf-8")) == audit_order):
            return previous
        raise RuntimeError("existing prepared run is incompatible; choose a fresh run ID")
    atomic_write_text(run_dir / "schedule.jsonl", "".join(canonical_json(row) + "\n" for row in rendered_rows))
    atomic_write_json(run_dir / "audit" / "review_order.json", audit_order)
    _snapshot_files(run_dir)

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


IDENTITY_FIELDS = (
    "response_id", "experiment", "task", "wording", "evidence", "order", "cue", "replicate", "block",
    "within_block", "generation_order", "seed", "raw_prompt",
)


def _validate_freeze(run_dir: Path, manifest: dict[str, Any], *, check_live: bool = True) -> list[dict[str, Any]]:
    """Validate all immutable inputs and the complete rendered schedule."""
    schedule = _schedule(run_dir)
    if len(schedule) != 144 or len({row.get("response_id") for row in schedule}) != 144:
        raise RuntimeError("frozen schedule must contain 144 unique response identities")
    if len({row.get("seed") for row in schedule}) != 144:
        raise RuntimeError("frozen schedule contains duplicate seeds")
    if sorted(row.get("generation_order") for row in schedule) != list(range(144)):
        raise RuntimeError("frozen schedule generation order is invalid")
    if _hash_value(schedule) != manifest.get("rendered_schedule_hash"):
        raise RuntimeError("rendered schedule hash mismatch")
    if any(row.get("prompt", {}).get("tokenizer_hash") != manifest.get("tokenizer_hash") for row in schedule):
        raise RuntimeError("schedule tokenizer provenance mismatch")
    if any(row.get("prompt", {}).get("chat_template_hash") != manifest.get("chat_template_hash") for row in schedule):
        raise RuntimeError("schedule chat-template provenance mismatch")
    audit_order_path = run_dir / "audit" / "review_order.json"
    if not audit_order_path.is_file():
        raise RuntimeError("frozen audit order is missing")
    audit_order = json.loads(audit_order_path.read_text(encoding="utf-8"))
    if (_hash_value(audit_order) != manifest.get("audit_order_hash")
            or len(audit_order) != len(set(audit_order))
            or set(audit_order) != {row["response_id"] for row in schedule}):
        raise RuntimeError("frozen audit order mismatch")
    frozen_config = load_yaml(run_dir / "config_snapshot" / CONFIG_PATH.name)
    frozen_prompts = load_yaml(run_dir / "config_snapshot" / PROMPTS_PATH.name)
    frozen_profile = load_yaml(run_dir / "config_snapshot" / PROFILE_PATH.name)
    frozen_profile["profile_name"] = PROFILE_NAME
    if manifest.get("frozen_manifest_hash") != _hash_value(
        _frozen_core(frozen_config, frozen_prompts, frozen_profile)
    ):
        raise RuntimeError("manifest does not match its frozen config/profile snapshots")
    if manifest.get("sampling") != frozen_config.get("sampling") or manifest.get("engine") != frozen_profile:
        raise RuntimeError("manifest sampling/engine settings differ from the frozen snapshots")
    expected = build_schedule(frozen_config, frozen_prompts)
    if manifest.get("raw_schedule_hash") != _hash_value(expected):
        raise RuntimeError("raw schedule hash mismatch")
    by_id = {row["response_id"]: row for row in schedule}
    for raw in expected:
        frozen = by_id.get(raw["response_id"])
        if frozen is None or any(frozen.get(field) != raw.get(field) for field in IDENTITY_FIELDS):
            raise RuntimeError(f"frozen schedule identity/factor mismatch: {raw['response_id']}")
    expected_order = [row["response_id"] for row in expected]
    random.Random(derive_experiment_seed(frozen_config["master_seed"], "audit-order")).shuffle(expected_order)
    if audit_order != expected_order:
        raise RuntimeError("audit review order differs from the frozen randomized order")
    for name, digest in manifest.get("config_hashes", {}).items():
        path = run_dir / "config_snapshot" / name
        if not path.is_file() or compute_checksum(path.read_bytes()) != digest:
            raise RuntimeError(f"frozen snapshot mismatch: {name}")
    if check_live:
        config, prompts, profile = load_experiment_config()
        git = git_info(REPO_ROOT)
        if _hash_value(_frozen_core(config, prompts, profile)) != manifest.get("frozen_manifest_hash"):
            raise RuntimeError("live config/profile differs from the prepared freeze")
        if compute_checksum((REPO_ROOT / "uv.lock").read_bytes()) != manifest.get("lockfile_hash"):
            raise RuntimeError("live dependency lock differs from the prepared freeze")
        if not manifest.get("mock") and (git["dirty"] or git["commit_sha"] != manifest.get("git_commit")):
            raise RuntimeError("production run requires the clean prepared Git commit")
    return schedule


def _validate_tokenizer(tokenizer: Any, schedule: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    if not manifest["mock"] and getattr(tokenizer, "name_or_path", "") == SimpleTokenizer.name_or_path:
        raise RuntimeError("production run refuses mock tokenizer provenance")
    revision = manifest["tokenizer_revision"]
    for row in schedule:
        rendered = render_user_prompt(tokenizer, row["raw_prompt"], revision)
        frozen = row["prompt"]
        actual = {
            "rendered_prompt": rendered.rendered, "raw_token_count": rendered.raw_token_count,
            "rendered_token_count": rendered.rendered_token_count, "raw_hash": rendered.raw_hash,
            "rendered_hash": rendered.rendered_hash, "tokenizer_hash": rendered.tokenizer_hash,
            "chat_template_hash": rendered.chat_template_hash,
        }
        if actual != frozen:
            raise RuntimeError(f"rendered prompt/tokenizer differs from freeze: {row['response_id']}")
        if rendered.rendered_token_count + manifest["sampling"]["max_tokens"] > manifest["engine"]["max_model_len"]:
            raise RuntimeError(f"prompt no longer fits 8192-token context: {row['response_id']}")


def _validate_completion(completion: Completion, prompt_tokens: int, max_completion_tokens: int = 4096) -> None:
    if not isinstance(completion.text, str):
        raise RuntimeError("backend response text is malformed")
    if completion.finish_reason not in {"stop", "length"}:
        raise RuntimeError(f"backend finish reason is missing or unknown: {completion.finish_reason!r}")
    if not isinstance(completion.prompt_tokens, int) or completion.prompt_tokens != prompt_tokens:
        raise RuntimeError("backend prompt-token accounting mismatch")
    if completion.token_ids is None:
        raise RuntimeError("backend omitted exact output token IDs")
    if not isinstance(completion.token_ids, list) or any(not isinstance(token, int) for token in completion.token_ids):
        raise RuntimeError("backend output token IDs are malformed")
    if not isinstance(completion.completion_tokens, int) or completion.completion_tokens != len(completion.token_ids):
        raise RuntimeError("backend completion-token accounting mismatch")
    if completion.completion_tokens > max_completion_tokens:
        raise RuntimeError("backend exceeded the frozen 4096-token output cap")


def _attempt_count(run_dir: Path, response_id: str) -> int:
    return len(list((run_dir / "errors").glob(f"{response_id}__attempt*.json")))


def _save_attempt(run_dir: Path, response_id: str, exc: BaseException, *, interrupted: bool = False) -> int:
    folder = "interruptions" if interrupted else "errors"
    attempt = (
        len(list((run_dir / folder).glob(f"{response_id}__attempt*.json"))) + 1
        if interrupted else _attempt_count(run_dir, response_id) + 1
    )
    atomic_write_json(run_dir / folder / f"{response_id}__attempt{attempt}.json", {
        "response_id": response_id, "attempt": attempt, "at": _utc_now(),
        "type": type(exc).__name__, "message": str(exc), "technical": not interrupted,
    })
    return attempt


def _runtime_signature(mock: bool) -> dict[str, Any]:
    report = environment_report()
    return {"mock": mock, "python": platform.python_version(), "packages": report["packages"],
            "gpu": report["gpu"], "cuda_runtime": report["cuda_runtime"]}


def _reported_engine_settings(backend: Backend) -> dict[str, Any]:
    engine = getattr(backend, "engine", None)
    llm_engine = getattr(engine, "llm_engine", None)
    model_config = getattr(llm_engine, "model_config", None)
    reported: dict[str, Any] = {}
    for name in ("dtype", "quantization", "max_model_len", "seed", "generation_config"):
        value = getattr(model_config, name, None)
        if isinstance(value, str | int | float | bool):
            reported[name] = value
    return reported


def gpu_preflight(run_dir: Path, cache_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Follow-up-specific A100 gate; it never loads the historical 4096-context profile."""
    profile = manifest["engine"]
    if profile.get("max_model_len") != 8192 or manifest["sampling"].get("max_tokens") != 4096:
        raise RuntimeError("follow-up preflight requires the frozen 8192/4096 context settings")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("production requires Python 3.11")
    if not path_is_writable(cache_dir) or not path_is_writable(run_dir):
        raise RuntimeError("cache and run paths must exist and be writable")
    repo = REPO_ROOT.resolve()
    if (cache_dir.resolve() == repo or repo in cache_dir.resolve().parents
            or run_dir.resolve() == repo or repo in run_dir.resolve().parents):
        raise RuntimeError("production cache and run directories must be outside the checkout")
    report = environment_report()
    gpu = report["gpu"]
    failures = []
    if len(gpu) != 1 or "A100" not in gpu[0]["name"].upper():
        failures.append("exactly one visible A100")
    if not gpu or int(gpu[0]["memory_total_mib"]) < 38 * 1024:
        failures.append("at least 38 GiB A100 capacity")
    if not report["torch_cuda_available"]:
        failures.append("CUDA usable by torch")
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    wanted = {"torch", "vllm", "transformers"}
    pinned = {
        item["name"].lower(): item["version"] for item in lock["package"] if item["name"].lower() in wanted
    }
    installed = package_versions(tuple(sorted(wanted)))
    if set(pinned) != wanted:
        failures.append("GPU packages present in uv.lock")
    failures.extend(f"pinned {name}=={version}" for name, version in pinned.items() if installed[name] != version)
    processes: list[str] = []
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        processes = [
            line.strip() for line in query.stdout.splitlines()
            if line.strip() and line.split(",", 1)[0].strip() != str(os.getpid())
        ]
    if processes:
        failures.append("no competing GPU compute/model process")
    artifact = verify_target(cache_dir, profile, load_transformers=False)
    if any(artifact.get(k) != manifest["model_verification"].get(k) for k in
           ("repository", "revision", "resolved_size_bytes", "small_metadata_sha256")):
        failures.append("matching prepared model artifact")
    report.update({"checked_at": _utc_now(), "effective_engine_settings": profile,
                   "effective_sampling_settings": manifest["sampling"], "pinned_versions": pinned,
                   "installed_versions": installed, "gpu_compute_processes": processes, "failures": failures})
    atomic_write_json(run_dir / "manifests" / "gpu_preflight.json", report)
    if failures:
        raise RuntimeError("A100 preflight failed: " + ", ".join(failures))
    return report


def _archive_checkpoint(run_dir: Path, count: int) -> Path:
    target = run_dir / "checkpoints" / f"responses_{count:03d}.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        for path in sorted((run_dir / "responses").glob("*.json")):
            archive.add(path, arcname=f"responses/{path.name}", recursive=False)
    atomic_write_text(target.with_suffix(target.suffix + ".sha256"), compute_checksum(target.read_bytes()) + "\n")
    return target


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
    with ProcessLock(run_dir):
        manifest_path = run_dir / "prepare_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("prepare must complete before run")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if bool(manifest.get("mock")) != mock:
            raise RuntimeError("mock/production mode must match the prepared run")
        schedule = _validate_freeze(run_dir, manifest)
        records = _records(run_dir)
        expected = {row["response_id"]: row for row in schedule}
        for rid, record in records.items():
            if rid not in expected or any(record.get(field) != expected[rid].get(field) for field in IDENTITY_FIELDS):
                raise RuntimeError(f"response-to-schedule association mismatch: {rid}")
            if record.get("frozen_manifest_hash") != manifest["frozen_manifest_hash"]:
                raise RuntimeError(f"response manifest reference mismatch: {rid}")
            if record.get("rendered_schedule_hash") != manifest["rendered_schedule_hash"]:
                raise RuntimeError(f"response schedule reference mismatch: {rid}")
            if record.get("rendered_prompt_hash") != expected[rid]["prompt"]["rendered_hash"]:
                raise RuntimeError(f"response prompt reference mismatch: {rid}")
        if records and not resume:
            raise RuntimeError("records already exist; pass --resume to continue")
        pending_all = [row for row in schedule if row["response_id"] not in records]
        if not pending_all:
            return status_run(run_dir)
        exhausted = [row["response_id"] for row in pending_all if _attempt_count(run_dir, row["response_id"]) >= 2]
        if not mock:
            exhausted.extend(
                f"canary_{name}" for name in ("normal", "stress_4096")
                if _attempt_count(run_dir, f"canary_{name}") >= 2
            )
        if exhausted:
            raise RuntimeError("INCOMPLETE_TECHNICAL: bounded retry limit reached: " + ", ".join(exhausted))
        pending = pending_all[:limit]
        if not mock:
            gpu_preflight(run_dir, cache_dir, manifest)
            enable_offline_mode()
        tokenizer = tokenizer or (SimpleTokenizer() if mock else _load_tokenizer(cache_dir, manifest["engine"]))
        _validate_tokenizer(tokenizer, schedule, manifest)
        supplied_backend = backend is not None
        if backend is None:
            runtime = dict(manifest["engine"])
            if not mock:
                runtime["model"] = str(artifact_path(cache_dir, manifest["model"], manifest["model_revision"]))
                runtime["tokenizer"] = str(
                    artifact_path(cache_dir, manifest["tokenizer"], manifest["tokenizer_revision"])
                )
            backend = make_backend(runtime, mock=mock)
        session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{os.getpid()}"
        signature = _runtime_signature(mock)
        runtime_path = run_dir / "manifests" / "generation_runtime.json"
        if runtime_path.exists() and json.loads(runtime_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise RuntimeError("generation runtime differs from the frozen first session")
        if not runtime_path.exists():
            atomic_write_json(runtime_path, {"signature": signature, "captured_at": _utc_now()})
        session_path = run_dir / "manifests" / "sessions" / f"{session_id}.json"
        session = {"session_id": session_id, "started_at": _utc_now(), "status": "CANARIES",
                   "existing_at_start": len(records), "pending_at_start": len(pending_all),
                   "effective_engine_settings": manifest["engine"], "sampling": manifest["sampling"],
                   "engine_runtime_reported": _reported_engine_settings(backend)}
        atomic_write_json(session_path, session)
        # Production gets a normal canary and an isolated forced-length 4096-token stress canary.
        if not mock:
            canaries = [
                ("normal", "Reply with exactly: canary pass", {**manifest["sampling"], "max_tokens": 64}, False),
                ("stress_4096", "Continue emitting the lowercase letter a separated by spaces until stopped.",
                 {**manifest["sampling"], "max_tokens": 4096}, True),
            ]
            for name, text, parameters, require_cap in canaries:
                try:
                    rendered = render_user_prompt(tokenizer, text, manifest["tokenizer_revision"])
                    result = backend.generate(
                        [rendered.rendered], [derive_experiment_seed(0, "canary", name)], parameters
                    )
                    if len(result) != 1:
                        raise RuntimeError(f"{name} canary returned the wrong completion count")
                    _validate_completion(result[0], rendered.rendered_token_count, parameters["max_tokens"])
                    if not result[0].text.strip() or "\ufffd" in result[0].text:
                        raise RuntimeError(f"{name} canary output is empty or malformed")
                    if require_cap and (result[0].completion_tokens != 4096 or result[0].finish_reason != "length"):
                        raise RuntimeError("4096-token stress canary did not exercise the full length cap")
                    atomic_write_json(run_dir / "manifests" / "canaries" / f"{session_id}_{name}.json", {
                        "status": "PASS", "session_id": session_id, "kind": name, "parameters": parameters,
                        "completion_tokens": result[0].completion_tokens, "finish_reason": result[0].finish_reason,
                    })
                except Exception as exc:
                    attempt = _save_attempt(run_dir, f"canary_{name}", exc)
                    session.update({"status": "CANARY_ERROR", "finished_at": _utc_now(), "error": str(exc)})
                    atomic_write_json(session_path, session)
                    state = "INCOMPLETE_TECHNICAL" if attempt >= 2 else "TECHNICAL_ERROR_RETRYABLE"
                    raise RuntimeError(f"{state}: {name} canary attempt {attempt}: {exc}") from exc
        session["status"] = "RUNNING"
        atomic_write_json(session_path, session)
        for row in pending:
            started = time.monotonic()
            rendered = row["prompt"]["rendered_prompt"]
            try:
                results = [_mock_completion(row)] if mock and not supplied_backend else backend.generate(
                    [rendered], [row["seed"]], manifest["sampling"]
                )
                if len(results) != 1:
                    raise RuntimeError("backend returned the wrong completion count")
                completion = results[0]
                _validate_completion(completion, row["prompt"]["rendered_token_count"], 4096)
                assert completion.token_ids is not None
                ids_384 = list(completion.token_ids[:384])
                ids_1024 = list(completion.token_ids[:1024])
                decoded_384 = tokenizer.decode(ids_384, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                decoded_1024 = tokenizer.decode(ids_1024, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except (KeyboardInterrupt, SystemExit) as exc:
                _save_attempt(run_dir, row["response_id"], exc, interrupted=True)
                session.update({"status": "INTERRUPTED", "finished_at": _utc_now(), "response_id": row["response_id"]})
                atomic_write_json(session_path, session)
                raise
            except Exception as exc:
                attempt = _save_attempt(run_dir, row["response_id"], exc)
                session.update({"status": "RESPONSE_ERROR", "finished_at": _utc_now(),
                                "response_id": row["response_id"], "error": str(exc)})
                atomic_write_json(session_path, session)
                state = "INCOMPLETE_TECHNICAL" if attempt >= 2 else "TECHNICAL_ERROR_RETRYABLE"
                raise RuntimeError(f"{state}: {row['response_id']} attempt {attempt}: {exc}") from exc
            payload = {
                "schema_version": 2,
                "response_id": row["response_id"],
                "experiment": row["experiment"],
                "task": row["task"],
                "wording": row["wording"],
                "evidence": row["evidence"],
                "order": row["order"],
                "cue": row["cue"],
                "replicate": row["replicate"],
                "block": row["block"],
                "within_block": row["within_block"],
                "generation_order": row["generation_order"],
                "seed": row["seed"],
                "raw_prompt": row["raw_prompt"],
                "rendered_prompt": rendered,
                "prompt_token_count": completion.prompt_tokens,
                "output_token_ids": list(completion.token_ids),
                "prefix_384_token_ids": ids_384,
                "prefix_1024_token_ids": ids_1024,
                "response_text": completion.text,
                "decoded_384": decoded_384,
                "decoded_1024": decoded_1024,
                "completion_token_count": completion.completion_tokens,
                "finish_reason": completion.finish_reason,
                "duration_seconds": time.monotonic() - started,
                "generated_at": _utc_now(),
                "immutable_content_hash": compute_checksum(completion.text),
                "frozen_manifest_hash": manifest["frozen_manifest_hash"],
                "rendered_schedule_hash": manifest["rendered_schedule_hash"],
                "rendered_prompt_hash": row["prompt"]["rendered_hash"],
                "session_id": session_id,
            }
            atomic_write_json(
                run_dir / "responses" / f"{row['response_id']}.json", {**payload, "record_hash": _hash_value(payload)}
            )
            records[row["response_id"]] = {**payload, "record_hash": _hash_value(payload)}
            if len(records) % 48 == 0:
                _archive_checkpoint(run_dir, len(records))
        atomic_write_text(
            run_dir / "responses.jsonl",
            "".join(
                canonical_json(records[row["response_id"]]) + "\n" for row in schedule if row["response_id"] in records
            ),
        )
        session.update({"status": "COMPLETE" if len(records) == 144 else "LIMIT_REACHED",
                        "finished_at": _utc_now(), "written": len(pending)})
        atomic_write_json(session_path, session)
    return status_run(run_dir)


def status_run(run_dir: Path) -> dict[str, Any]:
    schedule, records = _schedule(run_dir), _records(run_dir)
    failed_ids = {
        path.name.split("__attempt", 1)[0] for path in (run_dir / "errors").glob("*__attempt*.json")
    } if (run_dir / "errors").exists() else set()
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
    if len(records) == len(schedule):
        state = "COMPLETE"
    elif counts["failed"]:
        state = "INCOMPLETE_TECHNICAL"
    else:
        state = "IN_PROGRESS"
    counts["missing"] = len(schedule) - len(records)
    return {"status": state, **counts}


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
    manifest = json.loads((run / "prepare_manifest.json").read_text(encoding="utf-8"))
    schedule = _validate_freeze(run, manifest, check_live=False)
    records = _records(run)
    expected = {row["response_id"]: row for row in schedule}
    for rid, record in records.items():
        if rid not in expected or any(record.get(field) != expected[rid].get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"archive response-to-schedule mismatch: {rid}")
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
    with ProcessLock(run_dir):
        schedule = _schedule(run_dir)
        records = _records(run_dir)
        responses = run_dir / "responses.jsonl"
        atomic_write_text(responses, "".join(
            canonical_json(records[row["response_id"]]) + "\n"
            for row in schedule if row["response_id"] in records
        ))
        return ExperimentAnalyzer(responses, labels, schedule_path=run_dir / "schedule.jsonl").analyze(
            run_dir / "reports"
        )


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
        "saved_384",
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
        order_path = run_dir / "audit" / "review_order.json"
        order = json.loads(order_path.read_text(encoding="utf-8")) if order_path.exists() else [r["response_id"] for r in schedule]
        if len(order) != len(set(order)) or set(order) != {row["response_id"] for row in schedule}:
            raise RuntimeError("frozen audit review order is invalid")
        by_id = {row["response_id"]: row for row in schedule}
        for rid in order:
            row = by_id[rid]
            record = records.get(row["response_id"])
            if not record:
                continue
            writer.writerow(
                {
                    "response_id": row["response_id"],
                    "immutable_content_hash": record["immutable_content_hash"],
                    "saved_384": record["decoded_384"],
                    "saved_1024": record["decoded_1024"],
                    "full_response": record["response_text"],
                }
            )
    html_path = target.with_suffix(".html")
    html_path.write_text(
        "<html><body><p>Condition metadata is intentionally hidden. Label each prefix using only that prefix; "
        "label full using full context. Preserve exact quotes and distinguish ever-asserted claims from "
        "claims retracted by the observed end.</p><table>"
        + "".join(
            "<tr><td>"
            + html.escape(row["response_id"])
            + "</td><td><h4>384 tokens</h4><pre>" + html.escape(row["saved_384"])
            + "</pre><h4>1024 tokens</h4><pre>" + html.escape(row["saved_1024"])
            + "</pre><h4>Full saved response</h4><pre>" + html.escape(row["full_response"]) + "</pre></td></tr>"
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
    required_bool = [
        f"{field}_{window}"
        for field in ("execution_claim", "unsupported_measurements", "ambiguous", "retracted")
        for window in ("384", "1024", "full")
    ]
    required_columns = {"response_id", "immutable_content_hash", "saved_384", "saved_1024", "full_response", *required_bool,
                        *(f"evidence_quote_{window}" for window in ("384", "1024", "full"))}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError("label import is missing required outcome, ambiguity, retraction, quote, or immutable text columns")
    allowed = TRUE_VALUES | FALSE_VALUES
    for row in rows:
        rid = row["response_id"]
        if rid not in records:
            raise ValueError(f"{rid}: response has not been collected")
        if row.get("immutable_content_hash") != records[rid]["immutable_content_hash"]:
            raise ValueError(f"{rid}: immutable content hash mismatch")
        for column, expected_text in (("saved_384", records[rid]["decoded_384"]),
                                      ("saved_1024", records[rid]["decoded_1024"]),
                                      ("full_response", records[rid]["response_text"])):
            if row.get(column) != expected_text:
                raise ValueError(f"{rid}: immutable {column} mismatch")
        for key in required_bool:
            if str(row.get(key) or "").strip().lower() not in allowed:
                raise ValueError(f"{rid}: required label {key} is blank or invalid")
        for key, value in row.items():
            if (
                key.startswith(("execution_claim_", "unsupported_measurements_", "ambiguous_", "retracted_"))
                and str(value or "").strip().lower() not in allowed
            ):
                raise ValueError(f"{rid}: invalid label {key}")
            if (
                key.startswith("evidence_quote_")
                and (
                    _label_bool(row.get(f"execution_claim_{key.rsplit('_', 1)[-1]}")) is True
                    or _label_bool(row.get(f"unsupported_measurements_{key.rsplit('_', 1)[-1]}")) is True
                )
            ):
                window = key.rsplit("_", 1)[-1]
                view = records[rid]["response_text"] if window == "full" else records[rid][f"decoded_{window}"]
                if not str(value or "").strip() or str(value) not in view:
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
