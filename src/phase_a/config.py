"""Configuration loading and immutable snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .storage import atomic_write_text, compute_checksum

REPO_ROOT = Path(__file__).resolve().parents[2]


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_StrictConfig):
    repository: str
    release: str
    revision: str
    subsets: list[str]


class SelectionConfig(_StrictConfig):
    subject_model: str
    seed: int
    match_rate_min: float
    match_rate_max: float
    max_prompt_tokens: int
    naturalness_elo_min: float
    harmfulness_elo_max_exclusive: float
    transcripts_min: int
    primary_count: int = Field(gt=0, le=8)


class GenerationConfig(_StrictConfig):
    temperature: float
    top_p: float
    max_tokens: int
    reasoning: bool


class SamplingConfig(_StrictConfig):
    smoke_per_candidate: int = Field(gt=0, le=10)
    reproduction_per_candidate: int = Field(gt=0, le=30)
    max_reproduction_candidates: int = Field(gt=0, le=6)
    maximum_target_outputs: int = Field(gt=0, le=260)


class ReliabilityConfig(_StrictConfig):
    agreement_min: float
    sensitivity_min: float
    specificity_min: float


class AuditConfig(_StrictConfig):
    smoke_examples_per_candidate: int = Field(gt=0, le=10)
    include_invalid: bool


class AdvancementConfig(_StrictConfig):
    positive_min: int
    positive_max: int
    valid_min: int
    judged_min: int


class PhaseConfig(_StrictConfig):
    schema_version: int
    dataset: DatasetConfig
    selection: SelectionConfig
    generation: GenerationConfig
    sampling: SamplingConfig
    reliability: ReliabilityConfig
    audit: AuditConfig
    advancement: AdvancementConfig
    offline_by_default: bool


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def load_phase_config() -> dict[str, Any]:
    raw = load_yaml(REPO_ROOT / "configs" / "phase_a.yaml")
    config = PhaseConfig.model_validate(raw)
    expected = config.selection.primary_count * config.sampling.smoke_per_candidate + (
        config.sampling.max_reproduction_candidates * config.sampling.reproduction_per_candidate
    )
    if expected != config.sampling.maximum_target_outputs:
        raise ValueError(
            "sampling.maximum_target_outputs must exactly equal primary smoke plus maximum reproduction outputs"
        )
    return config.model_dump(mode="python")


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
