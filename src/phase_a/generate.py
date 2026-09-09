"""Balanced, deterministic, resumable target generation."""

from __future__ import annotations

import csv
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import artifact_path, load_phase_config, load_profile
from .inference import Backend, derive_seed, make_backend
from .run import load_manifest
from .schemas import GenerationRecord
from .storage import JSONLStorage


def balanced_schedule(candidates: list[dict[str, str]], samples: int, run_id: str) -> list[tuple[dict[str, str], int]]:
    schedule: list[tuple[dict[str, str], int]] = []
    for index in range(samples):
        batch = list(candidates)
        random.Random(derive_seed(run_id, "balanced-batch", index)).shuffle(batch)
        schedule.extend((candidate, index) for candidate in batch)
    return schedule


def _load_candidates(run_dir: Path, split: str) -> list[dict[str, str]]:
    source = run_dir / "selections" / "selected_candidates.csv"
    if split == "reproduction":
        advanced = run_dir / "selections" / "advanced_candidates.csv"
        if not advanced.exists():
            raise FileNotFoundError("run judge-smoke first; advanced_candidates.csv is missing")
        source = advanced
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    prompt_path = run_dir / "prompts" / "rendered_prompts.jsonl"
    prompts = {row["pattern_id"]: row for row in map(json.loads, prompt_path.read_text(encoding="utf-8").splitlines())}
    for row in rows:
        row.update(prompts[row["pattern_id"]])
    return rows


def count_target_outputs(run_dir: Path) -> int:
    count = 0
    for split in ("smoke", "reproduction"):
        count += len(
            JSONLStorage(run_dir / "responses" / f"responses_{split}.jsonl", GenerationRecord).load_valid_records()
        )
    return count


def run_generation(
    run_dir: Path,
    split: str,
    *,
    profile_name: str,
    backend: Backend | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    mock: bool = False,
    cache_dir: Path | None = None,
) -> dict[str, int]:
    if split not in {"smoke", "reproduction"}:
        raise ValueError("split must be smoke or reproduction")
    phase = load_phase_config()
    profile = load_profile(profile_name)
    manifest = load_manifest(run_dir)
    candidates = _load_candidates(run_dir, split)
    per_candidate = phase["sampling"]["smoke_per_candidate" if split == "smoke" else "reproduction_per_candidate"]
    schedule = balanced_schedule(candidates, per_candidate, run_dir.name)
    storage = JSONLStorage(run_dir / "responses" / f"responses_{split}.jsonl", GenerationRecord)
    existing = storage.valid_identities()
    pending = []
    for candidate, index in schedule:
        stub = GenerationRecord.model_construct(
            run_id=run_dir.name, split=split, pattern_id=candidate["pattern_id"], sample_index=index
        )
        if storage.identity(stub) not in existing:
            pending.append((candidate, index))
    if existing and not resume:
        raise RuntimeError("records already exist; pass --resume to continue safely")
    if limit is not None:
        pending = pending[:limit]
    projected = count_target_outputs(run_dir) + len(pending)
    if projected > phase["sampling"]["maximum_target_outputs"]:
        raise RuntimeError(f"Phase A output cap would be exceeded: {projected}")
    if dry_run:
        return {"existing": len(existing), "pending": len(pending), "written": 0}
    if not mock and backend is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required for local production inference")
        runtime_profile = dict(profile)
        runtime_profile["model"] = str(artifact_path(cache_dir, profile["model"], profile["revision"]))
        runtime_profile["tokenizer"] = str(
            artifact_path(
                cache_dir,
                profile.get("tokenizer", profile["model"]),
                profile.get("tokenizer_revision", profile["revision"]),
            )
        )
        backend = make_backend(runtime_profile)
    backend = backend or make_backend(profile, mock=True)
    parameters = phase["generation"]
    written = 0
    for candidate, index in pending:
        seed = derive_seed(run_dir.name, candidate["pattern_id"], index)
        start = time.monotonic()
        error_type = error_message = None
        try:
            completion = backend.generate([candidate["rendered_prompt"]], [seed], parameters)[0]
            valid = bool(completion.text.strip()) and completion.finish_reason in {"stop", "length"}
        except Exception as exc:
            from .inference import Completion

            completion = Completion("", None, 0, 0)
            valid, error_type, error_message = False, type(exc).__name__, str(exc)[:1000]
        record = GenerationRecord(
            run_id=run_dir.name,
            commit_sha=manifest.git_commit_sha,
            experiment_mode=manifest.experiment_mode,
            model_profile=profile_name,
            split=split,
            pattern_id=candidate["pattern_id"],
            behavior_id=candidate["behavior_id"],
            sample_index=index,
            seed=seed,
            raw_user_prompt=candidate["raw_user_text"],
            rendered_prompt_hash=candidate["rendered_prompt_hash"],
            response_text=completion.text,
            finish_reason=completion.finish_reason,
            prompt_token_count=completion.prompt_tokens,
            completion_token_count=completion.completion_tokens,
            model=profile["model"],
            model_revision=profile["revision"],
            tokenizer=profile.get("tokenizer", profile["model"]),
            tokenizer_revision=profile.get("tokenizer_revision", profile["revision"]),
            repository_revision=profile["revision"],
            chat_template_hash=candidate["chat_template_hash"],
            quantization_metadata=profile.get("quantization_metadata", {}),
            temperature=parameters["temperature"],
            top_p=parameters["top_p"],
            max_tokens=parameters["max_tokens"],
            reasoning_setting=parameters["reasoning"],
            generation_timestamp=datetime.now(UTC),
            generation_duration=time.monotonic() - start,
            validity_status=valid,
            error_type=error_type,
            error_message=error_message,
        )
        storage.append_record(record)
        written += 1
    return {"existing": len(existing), "pending": len(pending), "written": written}
