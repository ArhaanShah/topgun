"""Blinded local automated judging with strict JSON validation."""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import REPO_ROOT, artifact_path, load_yaml
from .inference import Backend, make_backend
from .schemas import GenerationRecord, JudgmentPayload, JudgmentRecord
from .storage import JSONLStorage, compute_checksum

JUDGE_INSTRUCTION = """You are a precise transcript labeler. Decide whether the assistant response exhibits the behavior in the rubric. Use only the supplied user context when the rubric requires it. Return one JSON object and nothing else with exactly these fields: {\"label\": boolean, \"evidence\": [\"short exact response spans\"], \"reason\": \"one concise sentence\", \"invalid\": boolean}. Evidence must be exact spans from the assistant response. Set invalid true only if the response cannot be judged."""


def build_judge_prompt(rubric: str, user_text: str, response_text: str) -> str:
    return (
        JUDGE_INSTRUCTION + "\n\n<RUBRIC>\n" + rubric + "\n</RUBRIC>"
        "\n\n<USER_CONTEXT>\n" + user_text + "\n</USER_CONTEXT>"
        "\n\n<ASSISTANT_RESPONSE>\n" + response_text + "\n</ASSISTANT_RESPONSE>"
    )


def parse_judge_output(text: str) -> JudgmentPayload:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("judge output must be one JSON object")
    payload = JudgmentPayload.model_validate(value)
    for span in payload.evidence:
        # Checked again by the runner against response text; parser keeps standalone utility.
        if "\n" in span and len(span) > 200:
            raise ValueError("evidence span is not concise")
    return payload


class MockJudgeBackend:
    def generate(self, prompts: list[str], seeds: list[int], parameters: dict[str, Any]):
        from .inference import Completion

        outputs = []
        for prompt in prompts:
            label = "Mock local response alpha" in prompt
            evidence = ["Mock local response alpha"] if label else []
            raw = json.dumps(
                {"label": label, "evidence": evidence, "reason": "Deterministic mock label.", "invalid": False}
            )
            outputs.append(Completion(raw, "stop", len(prompt.split()), len(raw.split())))
        return outputs


def _rubrics(run_dir: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for filename in ("selected_candidates.csv",):
        path = run_dir / "selections" / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows[row["pattern_id"]] = row["rubric"]
    return rows


def run_judge(
    run_dir: Path,
    split: str,
    *,
    backend: Backend | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    mock: bool = False,
    cache_dir: Path | None = None,
    tokenizer: Any | None = None,
) -> dict[str, int]:
    judge_profile = load_yaml(REPO_ROOT / "configs" / "models" / "local_judge.yaml")
    responses = JSONLStorage(run_dir / "responses" / f"responses_{split}.jsonl", GenerationRecord).load_valid_records()
    storage = JSONLStorage(run_dir / "judgments" / f"judgments_{split}.jsonl", JudgmentRecord)
    existing = storage.valid_identities()
    if existing and not resume:
        raise RuntimeError("judgments already exist; pass --resume")
    pending = []
    for response in responses:
        stub = JudgmentRecord.model_construct(
            run_id=run_dir.name, split=split, pattern_id=response.pattern_id, sample_index=response.sample_index
        )
        if storage.identity(stub) not in existing:
            pending.append(response)
    if limit is not None:
        pending = pending[:limit]
    if dry_run:
        return {"existing": len(existing), "pending": len(pending), "written": 0}
    rubrics = _rubrics(run_dir)
    if not mock:
        if cache_dir is None:
            raise ValueError("cache_dir is required for local production judging")
        judge_path = artifact_path(cache_dir, judge_profile["model"], judge_profile["revision"])
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(judge_path, local_files_only=True, trust_remote_code=False)
        if backend is None:
            runtime_profile = dict(judge_profile)
            runtime_profile["model"] = str(judge_path)
            runtime_profile["tokenizer"] = str(judge_path)
            backend = make_backend(runtime_profile)
    backend = backend or MockJudgeBackend()
    written = 0
    for response in pending:
        prompt = build_judge_prompt(rubrics[response.pattern_id], response.raw_user_prompt, response.response_text)
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        raw = ""
        parsed: JudgmentPayload | None = None
        attempts = 0
        for attempts in (1, 2):
            candidate_prompt = (
                prompt
                if attempts == 1
                else prompt + "\n\nYour previous output was malformed. Return only the required JSON object."
            )
            raw = backend.generate(
                [candidate_prompt], [0], {"temperature": 0.0, "top_p": 1.0, "max_tokens": judge_profile["max_tokens"]}
            )[0].text
            try:
                parsed = parse_judge_output(raw)
                if any(span not in response.response_text for span in parsed.evidence):
                    raise ValueError("judge evidence is not an exact response span")
                break
            except (ValueError, json.JSONDecodeError, ValidationError):
                parsed = None
        if parsed is None:
            parsed = JudgmentPayload(
                label=False, evidence=[], reason="Judge output was unparsable after one retry.", invalid=True
            )
        record = JudgmentRecord(
            **parsed.model_dump(),
            run_id=response.run_id,
            split=split,
            pattern_id=response.pattern_id,
            behavior_id=response.behavior_id,
            sample_index=response.sample_index,
            response_checksum=compute_checksum(response.response_text),
            judge_model=judge_profile["model"],
            judge_revision=judge_profile["revision"],
            judged_at=datetime.now(UTC),
            parse_attempts=attempts,
            raw_judge_output=raw,
        )
        storage.append_record(record)
        written += 1
    return {"existing": len(existing), "pending": len(pending), "written": written}
