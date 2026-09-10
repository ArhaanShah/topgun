"""Blinded audit export and completed-label import."""

from __future__ import annotations

import csv
import hashlib
import html
import random
from pathlib import Path

from .config import load_phase_config
from .run import load_manifest
from .schemas import AuditLabel, GenerationRecord, JudgmentRecord
from .statistics import passes_reliability, reliability_metrics
from .storage import JSONLStorage, atomic_write_json, atomic_write_text

AUDIT_FIELDS = [
    "run_id",
    "split",
    "pattern_id",
    "behavior_id",
    "sample_index",
    "rubric",
    "user_context",
    "response_text",
    "human_label",
    "ambiguity_flag",
    "notes",
]


def _load_rubrics(run_dir: Path) -> dict[str, str]:
    output = {}
    for name in ("selected_candidates.csv",):
        path = run_dir / "selections" / name
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                output.update({row["pattern_id"]: row["rubric"] for row in csv.DictReader(handle)})
    return output


def select_audit_rows(
    responses: list[GenerationRecord],
    judgments: list[JudgmentRecord],
    seed: int | None = None,
) -> list[GenerationRecord]:
    phase = load_phase_config()
    seed = phase["selection"]["seed"] if seed is None else seed
    judgment_by_id = {(j.split, j.pattern_id, j.sample_index): j for j in judgments}
    chosen: dict[tuple[str, str, int], GenerationRecord] = {}
    groups: dict[tuple[str, str], list[GenerationRecord]] = {}
    for response in responses:
        groups.setdefault((response.split, response.pattern_id), []).append(response)
        judgment = judgment_by_id.get((response.split, response.pattern_id, response.sample_index))
        if not response.validity_status or judgment is None or judgment.invalid:
            chosen[(response.split, response.pattern_id, response.sample_index)] = response
    for (split, pattern), group in sorted(groups.items()):
        count = phase["audit"]["smoke_examples_per_candidate"] if split == "smoke" else 8
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}||{split}||{pattern}".encode()).digest()[:8], "big"))
        positives = [
            r
            for r in group
            if (j := judgment_by_id.get((split, pattern, r.sample_index))) and j.label and not j.invalid
        ]
        negatives = [
            r
            for r in group
            if (j := judgment_by_id.get((split, pattern, r.sample_index))) and not j.label and not j.invalid
        ]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        balanced = positives[: count // 2] + negatives[: count // 2]
        remaining = [r for r in group if r not in balanced]
        rng.shuffle(remaining)
        for response in (balanced + remaining)[:count]:
            chosen[(split, pattern, response.sample_index)] = response
    return [chosen[key] for key in sorted(chosen)]


def create_audit_package(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    phase = manifest.resolved_configuration
    responses = JSONLStorage(run_dir / "responses" / "responses_smoke.jsonl", GenerationRecord).load_valid_records()
    judgments = JSONLStorage(run_dir / "judgments" / "judgments_smoke.jsonl", JudgmentRecord).load_valid_records()
    expected = phase["selection"]["primary_count"] * phase["sampling"]["smoke_per_candidate"]
    if manifest.experiment_mode != "mock" and (len(responses) != expected or len(judgments) != expected):
        raise RuntimeError(f"smoke audit requires exactly {expected} responses and {expected} judgments")
    rows = select_audit_rows(responses, judgments)
    rubrics = _load_rubrics(run_dir)
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audit_dir / "human_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for response in rows:
            writer.writerow(
                {
                    "run_id": response.run_id,
                    "split": response.split,
                    "pattern_id": response.pattern_id,
                    "behavior_id": response.behavior_id,
                    "sample_index": response.sample_index,
                    "rubric": rubrics[response.pattern_id],
                    "user_context": response.raw_user_prompt,
                    "response_text": response.response_text,
                    "human_label": "",
                    "ambiguity_flag": "",
                    "notes": "",
                }
            )
    instructions = """# Phase A human audit

For each row, read only the rubric, user context, and response. Enter `true` or `false` in `human_label`, `true` or `false` in `ambiguity_flag`, and optional brief notes. Do not consult published rates, automated labels, pattern titles, or candidate ordering. Save as CSV without changing identifying fields or response text.
"""
    atomic_write_text(audit_dir / "audit_instructions.md", instructions)
    sections = []
    for index, response in enumerate(rows, 1):
        sections.append(
            f"<section><h2>Item {index}</h2><h3>Rubric</h3><pre>{html.escape(rubrics[response.pattern_id])}</pre>"
            f"<h3>User context</h3><pre>{html.escape(response.raw_user_prompt)}</pre>"
            f"<h3>Assistant response</h3><pre>{html.escape(response.response_text)}</pre>"
            "<p>Record the label in human_audit.csv.</p></section>"
        )
    atomic_write_text(
        audit_dir / "audit_review.html",
        "<!doctype html><meta charset='utf-8'><title>Blinded Phase A audit</title><h1>Blinded audit</h1>"
        + "".join(sections),
    )
    return audit_dir


def _write_manual_smoke_sheet(
    run_dir: Path,
    responses: list[GenerationRecord],
    rubrics: dict[str, str],
    supplied: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    path = run_dir / "audit" / "manual_label_required.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for response in sorted(responses, key=lambda item: (item.pattern_id, item.sample_index)):
            key = (response.split, response.pattern_id, str(response.sample_index))
            existing = supplied.get(key, {})
            writer.writerow(
                {
                    "run_id": response.run_id,
                    "split": response.split,
                    "pattern_id": response.pattern_id,
                    "behavior_id": response.behavior_id,
                    "sample_index": response.sample_index,
                    "rubric": rubrics[response.pattern_id],
                    "user_context": response.raw_user_prompt,
                    "response_text": response.response_text,
                    "human_label": existing.get("human_label", ""),
                    "ambiguity_flag": existing.get("ambiguity_flag", ""),
                    "notes": existing.get("notes", ""),
                }
            )


def evaluate_smoke_audit(run_dir: str | Path) -> dict[str, object]:
    """Persist the human-reliability gate and select reproduction candidates only after it passes."""
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    phase = manifest.resolved_configuration
    responses = JSONLStorage(run_dir / "responses" / "responses_smoke.jsonl", GenerationRecord).load_valid_records()
    judgments = JSONLStorage(run_dir / "judgments" / "judgments_smoke.jsonl", JudgmentRecord).load_valid_records()
    with (run_dir / "audit" / "human_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    supplied = {(row["split"], row["pattern_id"], row["sample_index"]): row for row in rows}
    human = {key: label for key, row in supplied.items() if (label := _parse_bool(row.get("human_label"))) is not None}
    auto = {(item.split, item.pattern_id, item.sample_index): item for item in judgments}
    pairs = []
    for key, label in human.items():
        normalized = (key[0], key[1], int(key[2]))
        if normalized in auto and not auto[normalized].invalid:
            pairs.append((auto[normalized].label, label))
    metrics = reliability_metrics(pairs)
    reliable = passes_reliability(metrics, phase["reliability"])
    all_human = len(responses) > 0 and all(
        (response.split, response.pattern_id, str(response.sample_index)) in human for response in responses
    )
    passed = reliable or all_human
    advanced_ids: list[str] = []
    if passed:
        with (run_dir / "selections" / "selected_candidates.csv").open(newline="", encoding="utf-8") as handle:
            candidates = list(csv.DictReader(handle))
        advancement = phase["advancement"]
        for candidate in candidates:
            group = [response for response in responses if response.pattern_id == candidate["pattern_id"]]
            labels = []
            judged = 0
            for response in group:
                key = (response.split, response.pattern_id, response.sample_index)
                text_key = (response.split, response.pattern_id, str(response.sample_index))
                if text_key in human:
                    labels.append(human[text_key])
                    judged += 1
                elif reliable and key in auto and not auto[key].invalid:
                    labels.append(auto[key].label)
                    judged += 1
            positives = sum(labels)
            valid = sum(response.validity_status for response in group)
            if (
                advancement["positive_min"] <= positives <= advancement["positive_max"]
                and valid >= advancement["valid_min"]
                and judged >= advancement["judged_min"]
            ):
                advanced_ids.append(candidate["pattern_id"])
            if len(advanced_ids) == phase["sampling"]["max_reproduction_candidates"]:
                break
        advanced = [candidate for candidate in candidates if candidate["pattern_id"] in advanced_ids]
        path = run_dir / "selections" / "advanced_candidates.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]) if candidates else ["pattern_id"])
            writer.writeheader()
            writer.writerows(advanced)
    else:
        _write_manual_smoke_sheet(run_dir, responses, _load_rubrics(run_dir), supplied)
    report: dict[str, object] = {
        "status": "PASS" if passed else "FAIL",
        "reliability_pass": reliable,
        "all_smoke_human_labeled": all_human,
        "metrics": metrics,
        "advanced_pattern_ids": advanced_ids,
        "message": (
            "human smoke-audit gate passed"
            if passed
            else "reliability failed; label every row in audit/manual_label_required.csv and import it"
        ),
    }
    atomic_write_json(run_dir / "manifests" / "smoke_audit_gate.json", report)
    return report


def import_labels(run_dir: str | Path, labels_path: str | Path) -> int:
    run_dir, labels_path = Path(run_dir), Path(labels_path)
    expected_path = run_dir / "audit" / "human_audit.csv"
    with expected_path.open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    with labels_path.open(newline="", encoding="utf-8") as handle:
        supplied = list(csv.DictReader(handle))
    protected = (
        "run_id",
        "split",
        "pattern_id",
        "behavior_id",
        "sample_index",
        "rubric",
        "user_context",
        "response_text",
    )
    key_fields = ("split", "pattern_id", "sample_index")
    expected_by_key = {tuple(row[name] for name in key_fields): row for row in expected}
    supplied_by_key = {tuple(row[name] for name in key_fields): row for row in supplied}
    if len(supplied_by_key) != len(supplied) or not set(expected_by_key).issubset(supplied_by_key):
        raise ValueError("completed audit must contain every original row exactly once")
    rubrics = _load_rubrics(run_dir)
    valid_extra = {}
    for split in ("smoke", "reproduction"):
        records = JSONLStorage(
            run_dir / "responses" / f"responses_{split}.jsonl", GenerationRecord
        ).load_valid_records()
        for response in records:
            valid_extra[(response.split, response.pattern_id, str(response.sample_index))] = {
                "run_id": response.run_id,
                "split": response.split,
                "pattern_id": response.pattern_id,
                "behavior_id": response.behavior_id,
                "sample_index": str(response.sample_index),
                "rubric": rubrics[response.pattern_id],
                "user_context": response.raw_user_prompt,
                "response_text": response.response_text,
            }
    for key, completed in supplied_by_key.items():
        original = expected_by_key.get(key, valid_extra.get(key))
        if original is None or any(original[name] != completed.get(name) for name in protected):
            raise ValueError("audit identifying or blinded content fields were modified")
        AuditLabel.model_validate(
            {
                "run_id": completed["run_id"],
                "split": completed["split"],
                "pattern_id": completed["pattern_id"],
                "behavior_id": completed["behavior_id"],
                "sample_index": int(completed["sample_index"]),
                "human_label": _parse_bool(completed.get("human_label")),
                "ambiguity_flag": _parse_bool(completed.get("ambiguity_flag")),
                "notes": completed.get("notes", ""),
            }
        )
        if _parse_bool(completed.get("human_label")) is None:
            raise ValueError("every imported row requires a human_label")
    atomic_write_text(expected_path, labels_path.read_text(encoding="utf-8"))
    evaluate_smoke_audit(run_dir)
    return len(supplied)


def _parse_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--run", type=Path, required=True)
    importer = subparsers.add_parser("import-labels")
    importer.add_argument("--run", type=Path, required=True)
    importer.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(create_audit_package(args.run))
    else:
        print(f"Imported {import_labels(args.run, args.labels)} labels")


if __name__ == "__main__":
    main()
