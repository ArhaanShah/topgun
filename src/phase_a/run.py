"""Run directory creation and manifest helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from .config import REPO_ROOT, load_phase_config, load_profile, load_yaml, snapshot_configs
from .environment import environment_report, git_info
from .schemas import RunManifest
from .storage import atomic_write_json, atomic_write_text

RUN_SUBDIRS = (
    "config_snapshot",
    "manifests",
    "selections",
    "prompts",
    "responses",
    "judgments",
    "audit",
    "reports",
    "logs",
    "checkpoints",
)


def default_runs_dir() -> Path:
    return Path(os.environ.get("PHASE_A_RUNS_DIR", REPO_ROOT / ".phase_a_runs"))


def default_cache_dir() -> Path:
    return Path(os.environ.get("PHASE_A_CACHE_DIR", REPO_ROOT / ".phase_a_cache"))


def resolve_run_dir(runs_dir: str | Path | None, run_id: str | None, *, create: bool = False) -> Path:
    root = Path(runs_dir) if runs_dir else default_runs_dir()
    if not run_id:
        marker = root / "latest_run.txt"
        if not marker.exists():
            if not create:
                raise FileNotFoundError("no run ID supplied and no latest run exists")
            run_id = datetime.now(UTC).strftime("phase-a-%Y%m%dT%H%M%SZ")
        else:
            run_id = marker.read_text(encoding="utf-8").strip()
    run = root / run_id
    if create:
        for name in RUN_SUBDIRS:
            (run / name).mkdir(parents=True, exist_ok=True)
        atomic_write_text(root / "latest_run.txt", run_id + "\n")
    elif not run.exists():
        raise FileNotFoundError(f"run does not exist: {run}")
    return run


def initialize_manifest(
    run_dir: Path, profile_name: str, *, mock: bool, allow_dirty: bool, offline: bool
) -> RunManifest:
    phase = load_phase_config()
    profile = load_profile(profile_name)
    judge = load_yaml(REPO_ROOT / "configs" / "models" / "local_judge.yaml")
    git = git_info(REPO_ROOT)
    if git["dirty"] and not (allow_dirty or mock):
        raise RuntimeError("working tree is dirty; use a clean immutable commit for production")
    config_hashes = snapshot_configs(run_dir, profile_name)
    report = environment_report(run_dir / "manifests" / "environment.json")
    manifest = RunManifest(
        run_id=run_dir.name,
        created_at=datetime.now(UTC),
        git_commit_sha=git["commit_sha"],
        git_remote_url=git["remote_url"],
        git_dirty=bool(git["dirty"]) if not mock else False,
        experiment_mode="mock" if mock else profile["experiment_mode"],
        model_profile=profile_name,
        model=profile["model"],
        model_revision=profile["revision"],
        tokenizer=profile.get("tokenizer", profile["model"]),
        tokenizer_revision=profile.get("tokenizer_revision", profile["revision"]),
        dataset=phase["dataset"]["repository"],
        dataset_revision=phase["dataset"]["revision"],
        judge_model=judge["model"],
        judge_revision=judge["revision"],
        hardware={"gpu": report["gpu"], "system_ram_bytes": report["system_ram_bytes"]},
        software={"python": report["python"], **report["packages"]},
        quantization=profile.get("quantization_metadata", {}),
        generation_parameters=phase["generation"],
        hashes=config_hashes,
        seed_derivation="uint63(first 8 bytes of SHA256(run_id || pattern_id || sample_index))",
        offline=offline,
    )
    atomic_write_json(run_dir / "manifests" / "run_manifest.json", manifest)
    return manifest


def load_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
