"""Configuration loading and immutable snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .storage import atomic_write_text, compute_checksum

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def load_phase_config() -> dict[str, Any]:
    return load_yaml(REPO_ROOT / "configs" / "phase_a.yaml")


def load_profile(name: str) -> dict[str, Any]:
    aliases = {"fp8_offload": "qwen3_6_27b_fp8_offload", "awq_4bit": "qwen3_6_27b_awq"}
    path = REPO_ROOT / "configs" / "models" / f"{aliases.get(name, name)}.yaml"
    profile = load_yaml(path)
    profile["profile_name"] = name
    return profile


def artifact_path(cache_dir: str | Path, repository: str, revision: str) -> Path:
    return Path(cache_dir) / "models" / repository.replace("/", "--") / revision


def snapshot_configs(run_dir: Path, profile_name: str) -> dict[str, str]:
    destination = run_dir / "config_snapshot"
    suffix = "fp8_offload" if profile_name == "fp8_offload" else "awq"
    files = [
        REPO_ROOT / "configs" / "phase_a.yaml",
        REPO_ROOT / "configs" / "safety_exclusions.yaml",
        REPO_ROOT / "configs" / "models" / f"qwen3_6_27b_{suffix}.yaml",
        REPO_ROOT / "configs" / "models" / "local_judge.yaml",
    ]
    hashes: dict[str, str] = {}
    for source in files:
        text = source.read_text(encoding="utf-8")
        atomic_write_text(destination / source.name, text)
        hashes[source.name] = compute_checksum(text)
    return hashes
