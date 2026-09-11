"""Frozen 2x2 behavioral experiment lifecycle.

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

CONFIG_PATH = REPO_ROOT / "configs" / "understand_2x2.yaml"
PROMPTS_PATH = REPO_ROOT / "configs" / "understand_2x2_prompts.yaml"
PROFILE_NAME = "qwen3_6_27b_awq_a100"
PROFILE_PATH = REPO_ROOT / "configs" / "models" / f"{PROFILE_NAME}.yaml"
LOCK_PATH = "manifests/writer.lock"
BOOL_FIELDS = (
    "execution_claim",
    "specific_details",
    "unsupported_measurements",
    "gap_acknowledged",
    "retracted",
    "ambiguous",
)
WINDOWS = {"none", "by_384", "after_384", "crosses_384", "unclear"}


class SimpleTokenizer:
    """Small tokenizer implementing the production interface for CPU mocks."""

    name_or_path = "understand-2x2-mock-tokenizer"
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
            raise ValueError("2x2 prompts must be one user turn with thinking disabled")
        return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_value(value: Any) -> str:
    return compute_checksum(canonical_json(value))


def derive_experiment_seed(master_seed: int, namespace: str, *parts: object) -> int:
    payload = "||".join((str(master_seed), namespace, *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & 0x7FFFFFFF


def response_identity(experiment_id: str, paraphrase: str, evidence: int, code: int, replicate: int) -> str:
    payload = f"{experiment_id}||{paraphrase}||{evidence}||{code}||{replicate}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def load_experiment_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_yaml(CONFIG_PATH)
    prompts = load_yaml(PROMPTS_PATH)
    profile = load_profile(PROFILE_NAME)
    if config.get("profile") != PROFILE_NAME:
        raise ValueError(f"experiment profile must be {PROFILE_NAME}")
    if config.get("prediction_note", {}).get("frozen") is not True:
        raise ValueError("prediction_note.frozen must be true")
    if config.get("replicates_per_variant") != 10 or config.get("paraphrases") != ["P0", "P1"]:
        raise ValueError("the frozen design requires P0/P1 and ten replicates per variant")
    if config.get("sampling") != {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1024, "reasoning": False}:
        raise ValueError("sampling settings differ from the frozen 2x2 design")
    if profile.get("max_model_len") != 4096 or profile.get("generation_config") != "vllm":
        raise ValueError("A100 profile must use a 4096 context and vLLM generation defaults")
    return config, prompts, profile


def render_raw_variants(config: dict[str, Any], prompts: dict[str, Any]) -> list[dict[str, Any]]:
    separator = prompts["paragraph_separator"]
    rows = []
    for paraphrase in config["paraphrases"]:
        for evidence in (0, 1):
            for code in (0, 1):
                parts = [prompts["paraphrases"][paraphrase]]
                if code:
                    parts.extend((prompts["code_bridge"], prompts["code"]))
                parts.append(prompts["baseline_instruction"])
                if evidence:
                    parts.append(prompts["evidence_instruction"])
                rows.append(
                    {
                        "variant_id": f"{paraphrase}_E{evidence}_C{code}",
                        "paraphrase": paraphrase,
                        "evidence": evidence,
                        "code": code,
                        "raw_user_text": separator.join(parts),
                    }
                )
    if len(rows) != 8 or len({row["raw_user_text"] for row in rows}) != 8:
        raise AssertionError("the frozen design must render exactly eight distinct variants")
    return rows


def build_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = render_raw_variants(config, load_yaml(PROMPTS_PATH))
    by_id = {row["variant_id"]: row for row in variants}
    schedule: list[dict[str, Any]] = []
    seeds: set[int] = set()
    ids: set[str] = set()
    for replicate in range(config["replicates_per_variant"]):
        block = list(by_id)
        random.Random(derive_experiment_seed(config["master_seed"], "order", replicate)).shuffle(block)
        for within_block, variant_id in enumerate(block):
            variant = by_id[variant_id]
            factors = (variant["paraphrase"], variant["evidence"], variant["code"], replicate)
            seed = derive_experiment_seed(config["master_seed"], "sample", *factors)
            response_id = response_identity(config["experiment_id"], *factors)
            if seed in seeds or response_id in ids:
                raise AssertionError("response seeds and IDs must be unique")
            seeds.add(seed)
            ids.add(response_id)
            schedule.append(
                {
                    "response_id": response_id,
                    "variant_id": variant_id,
                    "paraphrase": factors[0],
                    "evidence": factors[1],
                    "code": factors[2],
                    "replicate": replicate,
                    "block": replicate,
                    "within_block": within_block,
                    "generation_order": len(schedule),
                    "seed": seed,
                }
            )
    if len(schedule) != 80:
        raise AssertionError("the frozen schedule must contain exactly 80 responses")
    for block in range(10):
        if len({row["variant_id"] for row in schedule if row["block"] == block}) != 8:
            raise AssertionError("each block must cover all eight variants")
    return schedule


def _require_explicit_paths(
    run_id: str | None, cache_dir: str | Path | None, runs_dir: str | Path | None
) -> tuple[Path, Path]:
    if not run_id or not str(run_id).strip():
        raise ValueError("--run-id is required and must be nonempty")
    if cache_dir is None or not str(cache_dir).strip() or runs_dir is None or not str(runs_dir).strip():
        raise ValueError("--cache-dir and --runs-dir are required and must be nonempty")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be one safe path component")
    return Path(cache_dir).expanduser().resolve(), Path(runs_dir).expanduser().resolve() / run_id


class ProcessLock(AbstractContextManager["ProcessLock"]):
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
    linux_boot_id = Path("/proc/sys/kernel/random/boot_id")
    if linux_boot_id.is_file():
        return linux_boot_id.read_text(encoding="utf-8").strip()
    import psutil

    return f"{platform.node()}:{psutil.boot_time():.6f}"


def _read_lock_payload(data: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("pid"), int):
            return payload
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    # Backward compatibility for locks produced before structured owner data.
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
    explicit = local / ".understand_2x2_provenance.json"
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
            "(or make 2x2-prepare ... DOWNLOAD=1)"
        )
    return {"revision": revision, "provenance_sources": sources}


def verify_target(cache_dir: Path, profile: dict[str, Any], *, load_transformers: bool = True) -> dict[str, Any]:
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
    from huggingface_hub import snapshot_download

    local = artifact_path(cache_dir, profile["model"], profile["revision"])
    snapshot_download(
        repo_id=profile["model"],
        revision=profile["revision"],
        local_dir=local,
        token=os.environ.get("HF_TOKEN"),
    )
    # A run-specific provenance marker does not replace Hub metadata, but makes the
    # exact download request explicit when local_dir metadata formats change.
    atomic_write_json(
        local / ".understand_2x2_provenance.json",
        {"repository": profile["model"], "revision": profile["revision"], "downloaded_at": _utc_now()},
    )


def _load_tokenizer(cache_dir: Path, profile: dict[str, Any]) -> Any:
    enable_offline_mode()
    from transformers import AutoTokenizer

    local = artifact_path(cache_dir, profile["tokenizer"], profile["tokenizer_revision"])
    return AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)


def _snapshot_files(run_dir: Path) -> dict[str, str]:
    destination = run_dir / "config_snapshot"
    paths = (CONFIG_PATH, PROMPTS_PATH, PROFILE_PATH, REPO_ROOT / "uv.lock")
    hashes = {}
    for source in paths:
        text = source.read_text(encoding="utf-8")
        atomic_write_text(destination / source.name, text)
        hashes[source.name] = compute_checksum(text)
    return hashes


def _frozen_core(config: dict[str, Any], prompts: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    return {"config": config, "prompts": prompts, "profile": profile, "sampling": config["sampling"]}


def prepare_experiment(
    run_id: str,
    cache_dir: Path,
    runs_dir: Path,
    *,
    download: bool = False,
    mock: bool = False,
) -> dict[str, Any]:
    config, prompt_config, profile = load_experiment_config()
    run_dir = runs_dir.resolve() / run_id
    frozen_hash = _hash_value(_frozen_core(config, prompt_config, profile))
    git = git_info(REPO_ROOT)
    lockfile_hash = compute_checksum((REPO_ROOT / "uv.lock").read_bytes())
    preparation_fingerprint = _hash_value(
        {"frozen_manifest_hash": frozen_hash, "git_commit_sha": git["commit_sha"], "lockfile_hash": lockfile_hash}
    )
    existing_path = run_dir / "manifests" / "manifest.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("preparation_fingerprint") == preparation_fingerprint and existing.get("run_id") == run_id:
            print(config["prediction_note"]["text"])
            return existing
        raise RuntimeError("run ID already exists with different frozen settings; choose a new run ID")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config_snapshot", "manifests", "private", "prompts", "responses", "errors", "audit", "analysis"):
        (run_dir / name).mkdir(exist_ok=True)
    with ProcessLock(run_dir):
        if git["dirty"] and not mock:
            raise RuntimeError("production prepare requires a clean Git tree and reviewed commit")
        if mock:
            tokenizer = SimpleTokenizer()
            artifact = {"mock": True, "repository": profile["model"], "revision": profile["revision"]}
        else:
            if download:
                _download_target(cache_dir, profile)
                command = [
                    sys.executable,
                    "-m",
                    "phase_a.understand_2x2",
                    "_verify-target",
                    "--cache-dir",
                    str(cache_dir),
                ]
                subprocess.run(command, cwd=REPO_ROOT, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
            artifact = verify_target(cache_dir, profile)
            tokenizer = _load_tokenizer(cache_dir, profile)
        rendered_rows = []
        for variant in render_raw_variants(config, prompt_config):
            rendered = render_user_prompt(tokenizer, variant["raw_user_text"], profile["tokenizer_revision"])
            if rendered.rendered_token_count + config["sampling"]["max_tokens"] > profile["max_model_len"]:
                raise RuntimeError(f"context overflow for {variant['variant_id']}; input truncation is forbidden")
            rendered_rows.append(
                {
                    **variant,
                    "rendered_prompt": rendered.rendered,
                    "raw_token_count": rendered.raw_token_count,
                    "rendered_token_count": rendered.rendered_token_count,
                    "raw_hash": rendered.raw_hash,
                    "rendered_hash": rendered.rendered_hash,
                    "tokenizer_hash": rendered.tokenizer_hash,
                    "chat_template_hash": rendered.chat_template_hash,
                }
            )
        schedule = build_schedule(config)
        prompt_by_id = {row["variant_id"]: row for row in rendered_rows}
        private_schedule = [{**row, "prompt": prompt_by_id[row["variant_id"]]} for row in schedule]
        audit_rng = random.Random(derive_experiment_seed(config["master_seed"], "audit"))
        audit_ids = [row["response_id"] for row in schedule]
        audit_rng.shuffle(audit_ids)
        hashes = _snapshot_files(run_dir)
        atomic_write_json(run_dir / "prompts" / "rendered_prompts.json", rendered_rows)
        atomic_write_json(run_dir / "private" / "schedule.json", private_schedule)
        atomic_write_json(run_dir / "private" / "audit_order.json", audit_ids)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": _utc_now(),
            "mock": mock,
            "status": "PREPARED_MOCK" if mock else "PREPARED",
            "experiment_id": config["experiment_id"],
            "frozen_manifest_hash": frozen_hash,
            "preparation_fingerprint": preparation_fingerprint,
            "lockfile_hash": lockfile_hash,
            "git_commit_sha": git["commit_sha"],
            "git_remote_url": git["remote_url"],
            "git_dirty": bool(git["dirty"]),
            "profile_name": PROFILE_NAME,
            "model": profile["model"],
            "model_revision": profile["revision"],
            "tokenizer": profile["tokenizer"],
            "tokenizer_revision": profile["tokenizer_revision"],
            "sampling": config["sampling"],
            "engine": profile,
            "prediction_note": config["prediction_note"],
            "schedule_count": len(schedule),
            "experimental_output_token_ceiling": len(schedule) * config["sampling"]["max_tokens"],
            "technical_canary_token_ceiling": config["sampling"]["max_tokens"],
            "paths": {
                "cache_dir": str(cache_dir.resolve()),
                "runs_dir": str(runs_dir.resolve()),
                "run_dir": str(run_dir.resolve()),
            },
            "artifact": artifact,
            "snapshot_hashes": hashes,
            "rendered_prompts_hash": _hash_value(rendered_rows),
            "private_schedule_hash": _hash_value(private_schedule),
            "seed_derivation": "31-bit SHA256(master_seed, namespace, P, E, C, replicate)",
        }
        atomic_write_json(existing_path, manifest)
    print("Frozen prediction note:", config["prediction_note"]["text"])
    print(f"Prepared {len(schedule)} responses in {run_dir} ({'MOCK' if mock else 'PRODUCTION'})")
    return manifest


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifests" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run is not prepared: {run_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config, prompts, profile = load_experiment_config()
    if manifest.get("frozen_manifest_hash") != _hash_value(_frozen_core(config, prompts, profile)):
        raise RuntimeError("current frozen files do not match the prepared manifest")
    for name, expected in manifest.get("snapshot_hashes", {}).items():
        snapshot = run_dir / "config_snapshot" / name
        if not snapshot.exists() or compute_checksum(snapshot.read_bytes()) != expected:
            raise RuntimeError(f"prepared configuration snapshot is missing or corrupt: {name}")
    rendered = json.loads((run_dir / "prompts" / "rendered_prompts.json").read_text(encoding="utf-8"))
    schedule = json.loads((run_dir / "private" / "schedule.json").read_text(encoding="utf-8"))
    if _hash_value(rendered) != manifest["rendered_prompts_hash"]:
        raise RuntimeError("rendered prompt snapshot hash mismatch")
    if _hash_value(schedule) != manifest["private_schedule_hash"]:
        raise RuntimeError("private schedule hash mismatch")
    return manifest


def _record_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "record_hash"}


def load_response_records(run_dir: Path, manifest: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    manifest = manifest or _load_manifest(run_dir)
    schedule = json.loads((run_dir / "private" / "schedule.json").read_text(encoding="utf-8"))
    expected = {row["response_id"]: row for row in schedule}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "responses").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"corrupt response record: {path.name}") from exc
        response_id = record.get("response_id")
        if path.stem != response_id or response_id not in expected:
            raise RuntimeError(f"unknown or mismatched response ID in {path.name}")
        if record.get("record_hash") != _hash_value(_record_payload(record)):
            raise RuntimeError(f"response checksum mismatch: {path.name}")
        row = expected[response_id]
        for field in ("variant_id", "paraphrase", "evidence", "code", "replicate", "seed", "generation_order"):
            if record.get(field) != row[field]:
                raise RuntimeError(f"frozen identity mismatch for {response_id}: {field}")
        if record.get("frozen_manifest_hash") != manifest["frozen_manifest_hash"]:
            raise RuntimeError(f"manifest reference mismatch for {response_id}")
        if response_id in records:
            raise RuntimeError(f"duplicate response ID: {response_id}")
        records[response_id] = record
    return records


def _mount_details(path: Path) -> dict[str, Any]:
    details: dict[str, Any] = {"path": str(path.resolve()), "device": None, "mount": None}
    try:
        details["device"] = os.stat(path).st_dev
        if shutil.which("findmnt"):
            result = subprocess.run(
                ["findmnt", "-T", str(path), "-no", "SOURCE,TARGET,FSTYPE"], capture_output=True, text=True, check=False
            )
            details["mount"] = result.stdout.strip()
    except OSError:
        pass
    return details


def gpu_preflight(run_dir: Path, cache_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("production requires Python 3.11")
    repo = REPO_ROOT.resolve()
    if cache_dir == repo or repo in cache_dir.parents or run_dir == repo or repo in run_dir.parents:
        raise RuntimeError("production cache and run paths must resolve outside the checkout")
    if not path_is_writable(cache_dir) or not path_is_writable(run_dir):
        raise RuntimeError("resolved cache and run paths must exist and be writable")
    git = git_info(REPO_ROOT)
    if git["dirty"] or git["commit_sha"] != manifest["git_commit_sha"]:
        raise RuntimeError("production run requires the clean prepared Git commit")
    report = environment_report()
    import torch

    visible_gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            visible_gpu.append({"index": index, "name": properties.name, "memory_total_bytes": properties.total_memory})
    failures = []
    if len(visible_gpu) != 1 or "A100" not in visible_gpu[0]["name"].upper():
        failures.append("exactly one visible A100")
    if not visible_gpu or visible_gpu[0]["memory_total_bytes"] < 38 * 1024**3:
        failures.append("at least 38 GiB visible GPU memory")
    if not report["torch_cuda_available"]:
        failures.append("CUDA usable by torch")
    processes: list[str] = []
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        processes = [
            line.strip()
            for line in proc.stdout.splitlines()
            if line.strip() and line.split(",", 1)[0].strip() != str(os.getpid())
        ]
    if processes:
        failures.append("no competing GPU compute/model process")
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    wanted_versions = {"torch", "vllm", "transformers"}
    pinned = {
        package["name"].lower(): package["version"]
        for package in lock["package"]
        if package["name"].lower() in wanted_versions
    }
    if set(pinned) != wanted_versions:
        raise RuntimeError("uv.lock omits a required GPU runtime package")
    versions = package_versions(tuple(pinned))
    failures.extend(f"pinned {name}=={version}" for name, version in pinned.items() if versions[name] != version)
    artifact = verify_target(cache_dir, load_profile(PROFILE_NAME), load_transformers=False)
    artifact_keys = (
        "repository",
        "revision",
        "path",
        "resolved_size_bytes",
        "small_metadata_sha256",
        "small_metadata_files",
    )
    if any(artifact.get(key) != manifest["artifact"].get(key) for key in artifact_keys):
        failures.append("matching prepared artifact manifest")
    storage = {
        "cache": _mount_details(cache_dir),
        "run": _mount_details(run_dir),
        "cache_free_bytes": shutil.disk_usage(cache_dir).free,
        "run_free_bytes": shutil.disk_usage(run_dir).free,
    }
    report.update(
        {
            "checked_at": _utc_now(),
            "versions": versions,
            "visible_gpu": visible_gpu,
            "gpu_compute_processes": processes,
            "storage": storage,
            "host_ram_available_bytes": __import__("psutil").virtual_memory().available,
            "multiprocessing": {
                "platform_start_method": __import__("multiprocessing").get_start_method(allow_none=True)
            },
            "quantization_kernel": os.environ.get("VLLM_USE_TRITON_AWQ", "vLLM-reported-at-engine-load"),
            "failures": failures,
        }
    )
    atomic_write_json(run_dir / "manifests" / "gpu_preflight.json", report)
    print("Cache mount:", storage["cache"])
    print("Run mount:", storage["run"])
    if failures:
        raise RuntimeError("A100 preflight failed: " + ", ".join(failures))
    return report


def _runtime_signature(mock: bool) -> dict[str, Any]:
    report = environment_report()
    return {
        "mock": mock,
        "python": platform.python_version(),
        "packages": report["packages"],
        "gpu": report["gpu"],
        "cuda_runtime": report["cuda_runtime"],
        "multiprocessing_start_method": __import__("multiprocessing").get_start_method(allow_none=True),
        "multiprocessing_environment": {
            "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }


def _reported_engine_settings(backend: Backend) -> dict[str, Any]:
    """Best-effort capture of fields exposed by the pinned vLLM engine."""
    engine = getattr(backend, "engine", None)
    llm_engine = getattr(engine, "llm_engine", None)
    model_config = getattr(llm_engine, "model_config", None)
    device_config = getattr(llm_engine, "device_config", None)
    reported = {}
    for name in ("dtype", "quantization", "max_model_len", "seed", "generation_config"):
        value = getattr(model_config, name, None)
        if value is not None and isinstance(value, str | int | float | bool):
            reported[name] = value
    device = getattr(device_config, "device", None)
    if device is not None:
        reported["device"] = str(device)
    return reported


def _check_runtime_session(run_dir: Path, signature: dict[str, Any]) -> None:
    path = run_dir / "manifests" / "generation_runtime.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise RuntimeError("GPU runtime differs from the first generation session; do not mix sessions")
    else:
        atomic_write_json(path, {"captured_at": _utc_now(), "signature": signature})


def _technical_attempts(run_dir: Path, response_id: str) -> int:
    return len(list((run_dir / "errors").glob(f"{response_id}__attempt*.json")))


def _save_error(run_dir: Path, response_id: str, exc: BaseException) -> int:
    attempt = _technical_attempts(run_dir, response_id) + 1
    atomic_write_json(
        run_dir / "errors" / f"{response_id}__attempt{attempt}.json",
        {
            "response_id": response_id,
            "attempt": attempt,
            "at": _utc_now(),
            "type": type(exc).__name__,
            "message": str(exc),
        },
    )
    return attempt


def _validate_completion(
    completion: Completion, expected_prompt_tokens: int, max_completion_tokens: int = 1024
) -> None:
    if not isinstance(completion.text, str):
        raise RuntimeError("backend response text is malformed")
    if not isinstance(completion.finish_reason, str) or not completion.finish_reason:
        raise RuntimeError("backend omitted a finish reason")
    if not isinstance(completion.prompt_tokens, int) or completion.prompt_tokens < 0:
        raise RuntimeError("backend prompt-token count is malformed")
    if not isinstance(completion.completion_tokens, int) or completion.completion_tokens < 0:
        raise RuntimeError("backend completion-token count is malformed")
    if completion.token_ids is None:
        raise RuntimeError("backend omitted exact output token IDs")
    if not isinstance(completion.token_ids, list) or any(not isinstance(token, int) for token in completion.token_ids):
        raise RuntimeError("backend output token IDs are malformed")
    if completion.completion_tokens != len(completion.token_ids):
        raise RuntimeError("backend completion-token accounting mismatch")
    if completion.completion_tokens > max_completion_tokens:
        raise RuntimeError("backend exceeded the frozen output-token cap")
    if completion.prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            f"backend prompt-token accounting mismatch: {completion.prompt_tokens} != {expected_prompt_tokens}"
        )


def _make_record(
    row: dict[str, Any], completion: Completion, duration: float, tokenizer: Any, manifest: dict[str, Any]
) -> dict[str, Any]:
    assert completion.token_ids is not None
    prefix_ids = list(completion.token_ids[:384])
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if manifest["mock"] and completion.completion_tokens <= 384:
        prefix_text = completion.text
    payload = {
        "schema_version": 1,
        "response_id": row["response_id"],
        "variant_id": row["variant_id"],
        "paraphrase": row["paraphrase"],
        "evidence": row["evidence"],
        "code": row["code"],
        "replicate": row["replicate"],
        "block": row["block"],
        "within_block": row["within_block"],
        "generation_order": row["generation_order"],
        "seed": row["seed"],
        "response_text": completion.text,
        "output_token_ids": list(completion.token_ids),
        "prefix_token_ids": prefix_ids,
        "prefix_text": prefix_text,
        "prompt_token_count": completion.prompt_tokens,
        "completion_token_count": completion.completion_tokens,
        "finish_reason": completion.finish_reason,
        "duration_seconds": duration,
        "empty_or_whitespace": not bool(completion.text.strip()),
        "generated_at": _utc_now(),
        "frozen_manifest_hash": manifest["frozen_manifest_hash"],
        "rendered_prompt_hash": row["prompt"]["rendered_hash"],
    }
    return {**payload, "record_hash": _hash_value(payload)}


def run_experiment(
    run_dir: Path,
    cache_dir: Path,
    *,
    resume: bool,
    mock: bool = False,
    backend: Backend | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(run_dir)
    if bool(manifest["mock"]) != bool(mock):
        raise RuntimeError("mock/production mode must match the prepared run")
    schedule = json.loads((run_dir / "private" / "schedule.json").read_text(encoding="utf-8"))
    records = load_response_records(run_dir, manifest)
    if records and not resume:
        raise RuntimeError("records already exist; pass --resume to continue safely")
    pending = [row for row in schedule if row["response_id"] not in records]
    print(f"{len(records)} existing, {len(pending)} pending")
    if not pending:
        return {"existing": len(records), "pending": 0, "written": 0, "status": "COMPLETE"}
    exhausted = [row["response_id"] for row in pending if _technical_attempts(run_dir, row["response_id"]) >= 2]
    if _technical_attempts(run_dir, "technical_canary") >= 2:
        exhausted.append("technical_canary")
    if exhausted:
        raise RuntimeError("INCOMPLETE_TECHNICAL: attempt cap reached for " + ", ".join(exhausted))
    with ProcessLock(run_dir):
        if not mock:
            gpu_preflight(run_dir, cache_dir, manifest)
            enable_offline_mode()
        signature = _runtime_signature(mock)
        _check_runtime_session(run_dir, signature)
        profile = load_profile(PROFILE_NAME)
        session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{os.getpid()}"
        session_path = run_dir / "manifests" / "sessions" / f"{session_id}.json"
        session = {
            "session_id": session_id,
            "started_at": _utc_now(),
            "status": "STARTING_ENGINE",
            "existing_at_start": len(records),
            "pending_at_start": len(pending),
            "runtime_signature": signature,
            "effective_engine_settings": manifest["engine"],
            "effective_sampling_settings": manifest["sampling"],
        }
        atomic_write_json(session_path, session)
        startup = time.monotonic()
        try:
            if tokenizer is None:
                tokenizer = SimpleTokenizer() if mock else _load_tokenizer(cache_dir, profile)
            if backend is None:
                runtime_profile = dict(profile)
                if not mock:
                    runtime_profile["model"] = str(artifact_path(cache_dir, profile["model"], profile["revision"]))
                    runtime_profile["tokenizer"] = str(
                        artifact_path(cache_dir, profile["tokenizer"], profile["tokenizer_revision"])
                    )
                backend = make_backend(runtime_profile, mock=mock)
        except Exception as exc:
            _save_error(run_dir, "engine_launch", exc)
            session.update({"status": "ENGINE_ERROR", "finished_at": _utc_now(), "error": str(exc)})
            atomic_write_json(session_path, session)
            raise
        startup_seconds = time.monotonic() - startup
        parameters = manifest["sampling"]
        session.update({"status": "RUNNING", "engine_startup_seconds": startup_seconds})
        session["engine_runtime_reported"] = _reported_engine_settings(backend)
        atomic_write_json(session_path, session)
        canary_path = run_dir / "manifests" / "canary.json"
        canary_text = "Write a numbered list of 500 distinct common English nouns, one per line, with no introduction."
        rendered = render_user_prompt(tokenizer, canary_text, profile["tokenizer_revision"])
        canary_seed = derive_experiment_seed(load_experiment_config()[0]["master_seed"], "canary")
        try:
            started = time.monotonic()
            canary = backend.generate([rendered.rendered], [canary_seed], parameters)[0]
            _validate_completion(canary, rendered.rendered_token_count)
            coherent = bool(canary.text.strip()) and not any(char == "\ufffd" for char in canary.text)
            if not coherent:
                raise RuntimeError("canary output is empty or contains decoding replacement characters")
            canary_record = {
                "status": "PASS",
                "session_id": session_id,
                "prompt": canary_text,
                "seed": canary_seed,
                "text": canary.text,
                "token_ids": canary.token_ids,
                "prompt_tokens": canary.prompt_tokens,
                "completion_tokens": canary.completion_tokens,
                "finish_reason": canary.finish_reason,
                "duration_seconds": time.monotonic() - started,
                "max_tokens": parameters["max_tokens"],
            }
            atomic_write_json(canary_path, canary_record)
            atomic_write_json(run_dir / "manifests" / "canaries" / f"{session_id}.json", canary_record)
        except Exception as exc:
            attempt = _save_error(run_dir, "technical_canary", exc)
            session.update({"status": "CANARY_ERROR", "finished_at": _utc_now(), "error": str(exc)})
            atomic_write_json(session_path, session)
            status = "INCOMPLETE_TECHNICAL" if attempt >= 2 else "TECHNICAL_ERROR_RETRYABLE"
            raise RuntimeError(f"{status}: technical canary attempt {attempt}: {exc}") from exc
        generation_seconds = 0.0
        written = 0
        for row in pending:
            response_id = row["response_id"]
            try:
                started = time.monotonic()
                completion = backend.generate([row["prompt"]["rendered_prompt"]], [row["seed"]], parameters)[0]
                duration = time.monotonic() - started
                _validate_completion(completion, row["prompt"]["rendered_token_count"])
                record = _make_record(row, completion, duration, tokenizer, manifest)
                atomic_write_json(run_dir / "responses" / f"{response_id}.json", record)
            except Exception as exc:
                attempt = _save_error(run_dir, response_id, exc)
                session.update(
                    {
                        "status": "RESPONSE_ERROR",
                        "finished_at": _utc_now(),
                        "response_id": response_id,
                        "error": str(exc),
                    }
                )
                atomic_write_json(session_path, session)
                status = "INCOMPLETE_TECHNICAL" if attempt >= 2 else "TECHNICAL_ERROR_RETRYABLE"
                raise RuntimeError(f"{status}: {response_id} attempt {attempt}: {exc}") from exc
            written += 1
            generation_seconds += duration
            completed = len(records) + written
            remaining = 80 - completed
            eta = generation_seconds / written * remaining if completed >= 8 else None
            print(
                f"completed={completed}/80 pending={remaining} last={duration:.2f}s "
                f"tokens={completion.completion_tokens} finish={completion.finish_reason} "
                f"generation_total={generation_seconds:.2f}s"
                + (f" eta_generation={eta:.2f}s startup={startup_seconds:.2f}s" if eta is not None else "")
            )
        atomic_write_text(
            run_dir / "responses" / "responses.jsonl",
            "".join(
                canonical_json(record) + "\n"
                for record in sorted(
                    load_response_records(run_dir, manifest).values(), key=lambda item: item["generation_order"]
                )
            ),
        )
        session.update(
            {
                "status": "COMPLETE",
                "finished_at": _utc_now(),
                "written": written,
                "experimental_generation_seconds": generation_seconds,
            }
        )
        atomic_write_json(session_path, session)
        return {"existing": len(records), "pending": len(pending), "written": written, "status": "COMPLETE"}


def export_audit(run_dir: Path) -> dict[str, Any]:
    with ProcessLock(run_dir):
        return _export_audit_unlocked(run_dir)


def _export_audit_unlocked(run_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(run_dir)
    records = load_response_records(run_dir, manifest)
    if len(records) != 80:
        raise RuntimeError(f"audit export requires 80 valid records; found {len(records)}")
    order = json.loads((run_dir / "private" / "audit_order.json").read_text(encoding="utf-8"))
    if len(order) != 80 or len(set(order)) != 80 or set(order) != set(records):
        raise RuntimeError("private audit order is corrupt or does not cover every response")
    audit = run_dir / "audit"
    columns = (
        "response_id",
        "immutable_content_hash",
        "response_text",
        "prefix_text",
        *BOOL_FIELDS,
        "first_claim_window",
        "evidence_quote",
        "notes",
    )
    with (audit / "human_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for response_id in order:
            record = records[response_id]
            writer.writerow(
                {
                    "response_id": response_id,
                    "immutable_content_hash": record["record_hash"],
                    "response_text": record["response_text"],
                    "prefix_text": record["prefix_text"],
                    **{field: "" for field in BOOL_FIELDS},
                    "first_claim_window": "",
                    "evidence_quote": "",
                    "notes": "",
                }
            )
    rubric = load_experiment_config()[0]["rubric"]
    examples = []
    for index, example in enumerate(rubric["synthetic_examples"], 1):
        expected = ", ".join(f"{key}={str(value).lower()}" for key, value in example.items() if key != "text")
        examples.append(f"{index}. “{example['text']}” — expected: `{expected}`")
    instructions = f"""# Human-label instructions

Label all 80 rows using the complete response context. This package is partially
blinded: condition mappings, prompts, seeds, finish reasons, and rates are hidden,
but wording in outputs may reveal condition. One researcher labels every row.

Primary `execution_claim`: {rubric["definition"]}

Exclusions: {rubric["exclusions"]}

Diagnostic labels:

{chr(10).join(f"- `{name}`: {definition}" for name, definition in rubric["diagnostic_definitions"].items())}

Synthetic rubric examples (interpret these before production labeling):

{chr(10).join(examples)}

Every Boolean cell must be the literal `true` or `false`. A primary positive needs
an exact short substring in `evidence_quote`; negatives use an empty quote.
`specific_details` and `retracted` imply a positive primary label. Use one of:
`none`, `by_384`, `after_384`, `crosses_384`, `unclear`. Negatives require `none`.
The prefix is a location aid only; interpret it in full-response context.
"""
    atomic_write_text(audit / "instructions.md", instructions)
    cards = []
    for index, response_id in enumerate(order, 1):
        record = records[response_id]
        cards.append(
            f"<article><h2>{index}. {html.escape(response_id)}</h2>"
            f"<h3>384-token prefix</h3><pre>{html.escape(record['prefix_text'])}</pre>"
            f"<h3>Full response</h3><pre>{html.escape(record['response_text'])}</pre></article>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>2x2 blinded audit</title>"
        "<style>body{max-width:1000px;margin:auto;font-family:sans-serif}article{border-top:2px solid #777}"
        "pre{white-space:pre-wrap;background:#eee;padding:1em}</style><h1>Partially blinded audit</h1>" + "".join(cards)
    )
    atomic_write_text(audit / "audit_review.html", document)
    return {"records": 80, "audit_dir": str(audit), "partial_blinding": True}


def _parse_bool(value: str, response_id: str, field: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"{response_id}: {field} must be literal true/false")
    return value == "true"


def validate_labels(run_dir: Path, labels_path: Path) -> list[dict[str, Any]]:
    records = load_response_records(run_dir)
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 80:
        raise ValueError(f"labels must contain exactly 80 rows; found {len(source_rows)}")
    seen: set[str] = set()
    validated = []
    for source in source_rows:
        response_id = source.get("response_id", "")
        if response_id not in records or response_id in seen:
            raise ValueError(f"labels contain an unknown or duplicate response ID: {response_id}")
        seen.add(response_id)
        record = records[response_id]
        if source.get("immutable_content_hash") != record["record_hash"]:
            raise ValueError(f"{response_id}: immutable content hash mismatch")
        if source.get("response_text") != record["response_text"] or source.get("prefix_text") != record["prefix_text"]:
            raise ValueError(f"{response_id}: immutable response content was changed")
        booleans = {field: _parse_bool(source.get(field, ""), response_id, field) for field in BOOL_FIELDS}
        window = source.get("first_claim_window", "")
        if window not in WINDOWS:
            raise ValueError(f"{response_id}: invalid first_claim_window")
        if not booleans["execution_claim"] and window != "none":
            raise ValueError(f"{response_id}: primary negatives require first_claim_window=none")
        if booleans["execution_claim"] and window == "none":
            raise ValueError(f"{response_id}: primary positives cannot use first_claim_window=none")
        if (booleans["specific_details"] or booleans["retracted"]) and not booleans["execution_claim"]:
            raise ValueError(f"{response_id}: specific_details/retracted imply execution_claim")
        quote = source.get("evidence_quote", "")
        if booleans["execution_claim"] and (not quote.strip() or quote not in record["response_text"]):
            raise ValueError(f"{response_id}: positive evidence_quote must be an exact response substring")
        if not booleans["execution_claim"] and quote.strip():
            raise ValueError(f"{response_id}: negatives must have an empty evidence_quote")
        validated.append(
            {
                "response_id": response_id,
                "immutable_content_hash": record["record_hash"],
                **booleans,
                "first_claim_window": window,
                "evidence_quote": quote,
                "notes": source.get("notes", ""),
                "audit_order": len(validated),
            }
        )
    if seen != set(records):
        raise ValueError("labels do not cover every response exactly once")
    return validated


def import_audit(run_dir: Path, labels_path: Path) -> Path:
    with ProcessLock(run_dir):
        return _import_audit_unlocked(run_dir, labels_path)


def _import_audit_unlocked(run_dir: Path, labels_path: Path) -> Path:
    rows = validate_labels(run_dir, labels_path)
    audit = run_dir / "audit"
    versions = sorted(audit.glob("imported_labels_v*.csv"))
    target = audit / f"imported_labels_v{len(versions) + 1}.csv"
    fields = list(rows[0])
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(
        audit / "active_labels.json",
        {"path": target.name, "sha256": compute_checksum(target.read_bytes()), "imported_at": _utc_now()},
    )
    return target


def mock_e2e(run_id: str, cache_dir: Path, runs_dir: Path) -> dict[str, Any]:
    prepare_experiment(run_id, cache_dir, runs_dir, mock=True)
    run_dir = runs_dir / run_id
    run_experiment(run_dir, cache_dir, resume=True, mock=True)
    export_audit(run_dir)
    source = run_dir / "audit" / "human_labels.csv"
    completed = run_dir / "audit" / "mock_completed_labels.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in BOOL_FIELDS:
            row[field] = "false"
        row["first_claim_window"] = "none"
    with completed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import_audit(run_dir, completed)
    from .understand_2x2_analysis import analyze

    analyze(run_dir)
    return {"status": "COMPLETE_MOCK", "run_dir": str(run_dir)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "export-audit", "import-audit", "analyze", "mock-e2e", "recover-lock"):
        item = sub.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.add_argument("--cache-dir", required=True)
        item.add_argument("--runs-dir", required=True)
        if name == "prepare":
            item.add_argument("--download", action="store_true")
            item.add_argument("--mock", action="store_true")
        elif name == "run":
            item.add_argument("--resume", action="store_true")
            item.add_argument("--mock", action="store_true")
        elif name == "import-audit":
            item.add_argument("--labels", required=True)
    verify = sub.add_parser("_verify-target")
    verify.add_argument("--cache-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_verify-target":
        print(json.dumps(verify_target(Path(args.cache_dir).resolve(), load_profile(PROFILE_NAME))))
        return 0
    cache_dir, run_dir = _require_explicit_paths(args.run_id, args.cache_dir, args.runs_dir)
    runs_dir = run_dir.parent
    if args.command == "prepare":
        prepare_experiment(args.run_id, cache_dir, runs_dir, download=args.download, mock=args.mock)
    elif args.command == "run":
        print(run_experiment(run_dir, cache_dir, resume=args.resume, mock=args.mock))
    elif args.command == "export-audit":
        print(export_audit(run_dir))
    elif args.command == "import-audit":
        print(import_audit(run_dir, Path(args.labels).expanduser().resolve()))
    elif args.command == "analyze":
        from .understand_2x2_analysis import analyze

        print(analyze(run_dir))
    elif args.command == "mock-e2e":
        print(mock_e2e(args.run_id, cache_dir, runs_dir))
    elif args.command == "recover-lock":
        print(recover_stale_lock(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
