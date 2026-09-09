"""Dataset preparation and exact prompt rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .config import REPO_ROOT, artifact_path, load_phase_config, load_profile
from .dataset import load_all_weirdchat, load_saved_weirdchat, normalize_patterns
from .inference import render_user_prompt
from .select_candidates import run_selection, write_prompt_review
from .storage import atomic_write_json, atomic_write_text, compute_checksum


class SimpleTokenizer:
    """Test-only renderer; production always uses the pinned official tokenizer."""

    name_or_path = "mock-tokenizer"
    chat_template = "<|user|>{{ content }}<|assistant|>"

    def __len__(self) -> int:
        return 1024

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"


def build_mock_frames(count: int = 12) -> dict[str, pl.DataFrame]:
    """Tiny complete-pipeline fixture used by local CPU smoke tests."""
    rows = []
    rubrics = []
    for index in range(count):
        behavior = f"benign-behavior-{index:02d}"
        rows.append(
            {
                "pattern_id": f"pattern-{index:02d}",
                "behavior_id": behavior,
                "subject_model": "qwen/qwen3.6-27b",
                "checkpoint": "mock",
                "representative_user_text": f"Tell me a benign fact number {index}.",
                "metrics": {"match_rate": 0.25},
                "elo": {"prompt_naturalness": {"elo": 1400.0}, "harmfulness": {"elo": 900.0}},
                "n_transcripts": 64,
                "title": f"Fixture {index}",
                "weirdchat_url": "https://example.invalid",
            }
        )
        rubrics.append(
            {
                "behavior_id": behavior,
                "transcript_rubric": {
                    "rubric_id": f"rubric-{index:02d}",
                    "text": "Label true only if the response contains the exact word alpha.",
                },
            }
        )
    return {"patterns": pl.DataFrame(rows), "rubrics": pl.DataFrame(rubrics), "prompts": pl.DataFrame()}


def prepare_run(
    run_dir: Path,
    profile_name: str,
    cache_dir: Path,
    *,
    mock_frames: dict[str, pl.DataFrame] | None = None,
    offline: bool = True,
    tokenizer: Any | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    phase = load_phase_config()
    profile = load_profile(profile_name)
    revision = profile.get("tokenizer_revision", profile["revision"])
    if tokenizer is None:
        if mock_frames is not None:
            tokenizer = SimpleTokenizer()
        else:
            from transformers import AutoTokenizer

            tokenizer_source = (
                artifact_path(cache_dir, profile.get("tokenizer", profile["model"]), revision)
                if offline
                else profile.get("tokenizer", profile["model"])
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                revision=revision,
                cache_dir=str(cache_dir / "huggingface"),
                local_files_only=offline,
                trust_remote_code=False,
            )
    if mock_frames is not None:
        frames = mock_frames
    elif offline:
        frames = load_saved_weirdchat(cache_dir, phase["dataset"]["revision"])
    else:
        frames = load_all_weirdchat(
            revision=phase["dataset"]["revision"],
            cache_dir=cache_dir / "huggingface",
            local_files_only=False,
        )
    normalized = normalize_patterns(
        frames["patterns"],
        frames.get("rubrics"),
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
    primary, reserve = run_selection(
        normalized,
        REPO_ROOT / "configs" / "safety_exclusions.yaml",
        run_dir / "selections",
        dataset_revision=phase["dataset"]["revision"],
        tokenizer_revision=revision,
        seed=phase["selection"]["seed"],
    )
    all_candidates = pl.concat([primary, reserve], how="vertical")
    prompt_records = []
    for row in all_candidates.to_dicts():
        rendered = render_user_prompt(tokenizer, str(row["representative_prompt"]), revision)
        prompt_records.append(
            {
                "pattern_id": str(row["pattern_id"]),
                "behavior_id": str(row["behavior_id"]),
                "raw_user_text": rendered.raw_user_text,
                "raw_token_count": rendered.raw_token_count,
                "rendered_token_count": rendered.rendered_token_count,
                "rendered_prompt": rendered.rendered,
                "raw_prompt_hash": rendered.raw_hash,
                "rendered_prompt_hash": rendered.rendered_hash,
                "tokenizer_hash": rendered.tokenizer_hash,
                "chat_template_hash": rendered.chat_template_hash,
                "system_message_count": 0,
                "prior_turn_count": 0,
                "trailing_assistant_content": False,
            }
        )
    lines = "\n".join(json.dumps(record, sort_keys=True, ensure_ascii=False) for record in prompt_records) + "\n"
    prompt_path = run_dir / "prompts" / "rendered_prompts.jsonl"
    atomic_write_text(prompt_path, lines)
    write_prompt_review(all_candidates, prompt_records, run_dir / "prompts" / "prompt_review.html")
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hashes"].update(
        {
            "rendered_prompts": compute_checksum(lines),
            "tokenizer": prompt_records[0]["tokenizer_hash"] if prompt_records else "",
            "chat_template": prompt_records[0]["chat_template_hash"] if prompt_records else "",
        }
    )
    atomic_write_json(manifest_path, manifest)
    for name in ("artifact_manifest.json", "preflight.json", "bootstrap_environment.json"):
        source = cache_dir / name
        if source.exists():
            atomic_write_text(run_dir / "manifests" / name, source.read_text(encoding="utf-8"))
    return primary, reserve
