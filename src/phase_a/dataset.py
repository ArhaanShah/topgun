"""Pinned WeirdChat loading and schema normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

DATASET_ID = "Transluce/WeirdChat"
DATASET_REVISION = "3460e41491308601cd68ee377011c7b7a8f0b4b3"


def load_weirdchat_subset(
    subset: str,
    revision: str = DATASET_REVISION,
    cache_dir: str | Path | None = None,
    *,
    local_files_only: bool = False,
) -> pl.DataFrame:
    """Load one pinned public subset; never uses a hosted inference service."""
    if local_files_only:
        from .environment import enable_offline_mode

        enable_offline_mode()
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        name=subset,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        split="train",
    )
    return pl.from_arrow(dataset.data.table)


def load_all_weirdchat(
    revision: str = DATASET_REVISION,
    cache_dir: str | Path | None = None,
    *,
    local_files_only: bool = False,
) -> dict[str, pl.DataFrame]:
    return {
        name: load_weirdchat_subset(name, revision, cache_dir, local_files_only=local_files_only)
        for name in ("patterns", "prompts", "rubrics")
    }


def load_saved_weirdchat(cache_dir: str | Path, revision: str = DATASET_REVISION) -> dict[str, pl.DataFrame]:
    """Load the explicit on-disk snapshots produced by download_artifacts.py."""
    from datasets import load_from_disk

    root = Path(cache_dir) / "datasets" / "Transluce--WeirdChat" / revision
    return {name: pl.from_arrow(load_from_disk(root / name).data.table) for name in ("patterns", "prompts", "rubrics")}


def _nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(key)
        else:
            try:
                value = value[key]
            except (KeyError, TypeError):
                return None
    return value


def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _rubric_text(value: Any) -> str | None:
    """Extract rubric text from both legacy strings and WeirdChat rubric structs."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        text = value.get("text")
        return str(text) if text else None
    return None


def normalize_patterns(
    patterns: pl.DataFrame,
    rubrics: pl.DataFrame | None = None,
    token_counter: Any | None = None,
) -> pl.DataFrame:
    """Convert the release schema (nested structs) into selector columns."""
    rubric_by_behavior: dict[str, str] = {}
    if rubrics is not None:
        for row in rubrics.to_dicts():
            behavior = str(_first(row, ("behavior_id", "id")) or "")
            rubric = _first(row, ("transcript_rubric", "rubric", "assistant_rubric"))
            rubric_text = _rubric_text(rubric)
            if behavior and rubric_text:
                rubric_by_behavior[behavior] = rubric_text

    normalized: list[dict[str, Any]] = []
    for row in patterns.to_dicts():
        prompt = _first(row, ("representative_user_text", "representative_prompt", "user_text"))
        behavior = str(row.get("behavior_id") or "")
        prompt_text = "" if prompt is None else str(prompt)
        token_count = (
            token_counter(prompt_text) if token_counter and prompt_text else row.get("representative_prompt_length")
        )
        metrics = row.get("metrics")
        elo = row.get("elo")
        normalized.append(
            {
                "pattern_id": row.get("pattern_id"),
                "behavior_id": behavior or None,
                "subject_model": row.get("subject_model"),
                "checkpoint": row.get("checkpoint"),
                "representative_prompt": prompt,
                "rubric": _rubric_text(_first(row, ("rubric", "transcript_rubric")))
                or rubric_by_behavior.get(behavior),
                "match_rate": row.get("match_rate")
                if row.get("match_rate") is not None
                else _nested(metrics, "match_rate"),
                "prompt_naturalness_elo": row.get("prompt_naturalness_elo")
                if row.get("prompt_naturalness_elo") is not None
                else _nested(elo, "prompt_naturalness", "elo"),
                "harmfulness_elo": row.get("harmfulness_elo")
                if row.get("harmfulness_elo") is not None
                else _nested(elo, "harmfulness", "elo"),
                "transcript_count": row.get("transcript_count")
                if row.get("transcript_count") is not None
                else row.get("n_transcripts"),
                "representative_prompt_length": token_count,
                "title": row.get("title"),
                "weirdchat_url": row.get("weirdchat_url"),
            }
        )
    return pl.DataFrame(normalized)
