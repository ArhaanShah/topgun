"""Deterministic, safety-scoped candidate selection and prompt review."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from .config import load_phase_config
from .storage import atomic_write_json, atomic_write_text, compute_checksum


def load_exclusions(path: str | Path) -> dict[str, str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("behaviors", data.get("exclusions", []))
    if isinstance(entries, dict):
        return {str(key): str(value) for key, value in entries.items()}
    output: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, dict):
            output[str(entry["behavior_id"])] = str(entry["reason"])
        else:
            output[str(entry)] = "configured safety exclusion"
    return output


def exclusion_reason(
    row: dict[str, Any], exclusions: dict[str, str], selection: dict[str, Any] | None = None
) -> str | None:
    selection = selection or load_phase_config()["selection"]
    required = ("pattern_id", "behavior_id", "representative_prompt", "rubric")
    missing = [name for name in required if row.get(name) is None or str(row.get(name)).strip() == ""]
    if missing:
        return "missing:" + ",".join(missing)
    behavior = str(row["behavior_id"])
    if behavior in exclusions:
        return f"safety:{exclusions[behavior]}"
    checks = (
        (row.get("subject_model") == selection["subject_model"], "subject_model"),
        (
            row.get("match_rate") is not None
            and selection["match_rate_min"] <= float(row["match_rate"]) <= selection["match_rate_max"],
            "match_rate",
        ),
        (
            row.get("representative_prompt_length") is not None
            and int(row["representative_prompt_length"]) <= selection["max_prompt_tokens"],
            "prompt_length",
        ),
        (
            row.get("prompt_naturalness_elo") is not None
            and float(row["prompt_naturalness_elo"]) >= selection["naturalness_elo_min"],
            "naturalness",
        ),
        (
            row.get("harmfulness_elo") is not None
            and float(row["harmfulness_elo"]) < selection["harmfulness_elo_max_exclusive"],
            "harmfulness",
        ),
        (
            row.get("transcript_count") is not None and int(row["transcript_count"]) >= selection["transcripts_min"],
            "transcript_count",
        ),
    )
    return next((reason for passed, reason in checks if not passed), None)


def partition_eligible(
    df: pl.DataFrame, exclusions: dict[str, str] | list[str], selection: dict[str, Any] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    mapping = (
        exclusions if isinstance(exclusions, dict) else {item: "configured safety exclusion" for item in exclusions}
    )
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in df.to_dicts():
        reason = exclusion_reason(row, mapping, selection)
        if reason is None:
            eligible.append(row)
        else:
            rejected.append(
                {"pattern_id": row.get("pattern_id"), "behavior_id": row.get("behavior_id"), "reason": reason}
            )
    return pl.DataFrame(eligible, schema=df.schema) if eligible else df.head(0), pl.DataFrame(rejected)


def apply_filters(
    df: pl.DataFrame, exclusions: dict[str, str] | list[str], selection: dict[str, Any] | None = None
) -> pl.DataFrame:
    return partition_eligible(df, exclusions, selection)[0]


def _shuffle_key(pattern_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}||{pattern_id}".encode()).hexdigest()


def ordered_candidates(df: pl.DataFrame, seed: int | None = None) -> pl.DataFrame:
    seed = load_phase_config()["selection"]["seed"] if seed is None else seed
    rows = sorted(df.to_dicts(), key=lambda row: (_shuffle_key(str(row["pattern_id"]), seed), str(row["pattern_id"])))
    seen: set[str] = set()
    unique = []
    for row in rows:
        behavior = str(row["behavior_id"])
        if behavior not in seen:
            seen.add(behavior)
            unique.append(row)
    return pl.DataFrame(unique, schema=df.schema) if unique else df.head(0)


def select_candidates(df: pl.DataFrame, seed: int | None = None, count: int | None = None) -> pl.DataFrame:
    selection = load_phase_config()["selection"]
    seed = selection["seed"] if seed is None else seed
    count = selection["primary_count"] if count is None else count
    ordered = ordered_candidates(df, seed)
    return ordered.head(count)


def write_prompt_review(candidates: pl.DataFrame, prompt_records: list[dict[str, Any]], output: Path) -> None:
    by_pattern = {record["pattern_id"]: record for record in prompt_records}
    rows = []
    for candidate in candidates.to_dicts():
        record = by_pattern[str(candidate["pattern_id"])]
        rows.append(
            "<section><h2>" + html.escape(str(candidate["pattern_id"])) + "</h2>"
            "<dl><dt>Raw/rendered tokens</dt><dd>"
            + str(record["raw_token_count"])
            + "/"
            + str(record["rendered_token_count"])
            + "</dd>"
            "<dt>System messages</dt><dd>0</dd><dt>Prior/few-shot turns</dt><dd>0</dd>"
            "<dt>Trailing assistant content</dt><dd>none</dd></dl>"
            "<h3>Raw user text</h3><pre>" + html.escape(record["raw_user_text"]) + "</pre>"
            "<h3>Complete rendered prompt</h3><pre>" + html.escape(record["rendered_prompt"]) + "</pre></section>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Phase A prompt review</title><h1>Prompt review</h1>"
        + "".join(rows)
    )
    atomic_write_text(output, document)


def run_selection(
    patterns_df: pl.DataFrame,
    exclusions_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_revision: str = "",
    tokenizer_revision: str = "",
    seed: int | None = None,
) -> pl.DataFrame:
    selection = load_phase_config()["selection"]
    seed = selection["seed"] if seed is None else seed
    exclusions = load_exclusions(exclusions_path)
    eligible, rejected = partition_eligible(patterns_df, exclusions, selection)
    primary = select_candidates(eligible, seed, selection["primary_count"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eligible.write_parquet(out / "eligible_patterns.parquet")
    primary.write_csv(out / "selected_candidates.csv")
    rejected.write_csv(out / "exclusions.csv")
    ordered_ids = ordered_candidates(eligible, seed).get_column("pattern_id").to_list()
    manifest = {
        "schema_version": 1,
        "dataset": "Transluce/WeirdChat",
        "dataset_revision": dataset_revision,
        "tokenizer_revision": tokenizer_revision,
        "selection_seed": seed,
        "shuffle_algorithm": "ascending SHA256(seed || pattern_id)",
        "subject_model": selection["subject_model"],
        "filters": {
            "match_rate": [selection["match_rate_min"], selection["match_rate_max"]],
            "max_prompt_tokens": selection["max_prompt_tokens"],
            "min_naturalness_elo": selection["naturalness_elo_min"],
            "max_harmfulness_elo_exclusive": selection["harmfulness_elo_max_exclusive"],
            "min_transcripts": selection["transcripts_min"],
            "one_pattern_per_behavior": True,
        },
        "safety_exclusions_sha256": compute_checksum(Path(exclusions_path).read_text(encoding="utf-8")),
        "eligible_pattern_ids": sorted(eligible.get_column("pattern_id").to_list()),
        "ordered_unique_pattern_ids": ordered_ids,
        "primary_pattern_ids": primary.get_column("pattern_id").to_list(),
    }
    atomic_write_json(out.parent / "manifests" / "selection_manifest.json", manifest)
    return primary
