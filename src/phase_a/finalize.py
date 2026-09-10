"""CPU-only audit reconciliation, qualification, and final reporting."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import yaml

from .audit import _parse_bool
from .run import load_manifest
from .schemas import GenerationRecord, JudgmentRecord
from .statistics import compute_wilson_interval, passes_reliability, qualifies_case, reliability_metrics
from .storage import JSONLStorage, atomic_write_text


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _consolidate(storage_paths: list[Path], schema: type, output: Path) -> list[Any]:
    records = []
    for path in storage_paths:
        records.extend(JSONLStorage(path, schema).load_valid_records())
    lines = [json.dumps(record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) for record in records]
    atomic_write_text(output, "\n".join(lines) + ("\n" if lines else ""))
    return records


def select_final_case_ids(results: list[dict[str, Any]], original_order: list[str], count: int = 4) -> list[str]:
    qualified = {row["pattern_id"] for row in results if row["qualifies"]}
    return [pattern_id for pattern_id in original_order if pattern_id in qualified][:count]


def finalize_run(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    responses = _consolidate(
        [run_dir / "responses" / f"responses_{split}.jsonl" for split in ("smoke", "reproduction")],
        GenerationRecord,
        run_dir / "responses.jsonl",
    )
    judgments = _consolidate(
        [run_dir / "judgments" / f"judgments_{split}.jsonl" for split in ("smoke", "reproduction")],
        JudgmentRecord,
        run_dir / "judgments.jsonl",
    )
    audit_path = run_dir / "audit" / "human_audit.csv"
    audit_rows = _csv_rows(audit_path)
    if audit_path.exists():
        atomic_write_text(run_dir / "human_audit.csv", audit_path.read_text(encoding="utf-8"))
    else:
        atomic_write_text(run_dir / "human_audit.csv", "status\nPENDING\n")
    human = {}
    systematic = set()
    for row in audit_rows:
        label = _parse_bool(row.get("human_label"))
        if label is not None:
            human[(row["split"], row["pattern_id"], int(row["sample_index"]))] = label
        if "SYSTEMATIC_RUBRIC_ERROR" in row.get("notes", "").upper():
            systematic.add(row["pattern_id"])
    auto = {(j.split, j.pattern_id, j.sample_index): j for j in judgments}
    reproduction = [r for r in responses if r.split == "reproduction"]
    groups: dict[str, list[GenerationRecord]] = {}
    for response in reproduction:
        groups.setdefault(response.pattern_id, []).append(response)
    candidates = _csv_rows(run_dir / "selections" / "selected_candidates.csv")
    selection_manifest = json.loads((run_dir / "manifests" / "selection_manifest.json").read_text(encoding="utf-8"))
    order = selection_manifest["ordered_unique_pattern_ids"]
    candidate_by_id = {row["pattern_id"]: row for row in candidates}
    results = []
    needs_manual: list[dict[str, Any]] = []
    for pattern_id in order:
        group = sorted(groups.get(pattern_id, []), key=lambda item: item.sample_index)
        if not group:
            continue
        pairs = []
        for response in group:
            key = (response.split, response.pattern_id, response.sample_index)
            if key in human and key in auto and not auto[key].invalid:
                pairs.append((auto[key].label, human[key]))
        metrics = reliability_metrics(pairs)
        reliable = passes_reliability(metrics) and pattern_id not in systematic
        human_count = sum((r.split, r.pattern_id, r.sample_index) in human for r in group)
        all_human = len(group) == 30 and human_count == 30
        valid = [r for r in group if r.validity_status]
        labels = []
        for response in valid:
            key = (response.split, response.pattern_id, response.sample_index)
            if key in human:
                labels.append(human[key])
            elif key in auto and not auto[key].invalid:
                labels.append(auto[key].label)
        positive = sum(labels)
        valid_count = len(valid)
        rate = positive / valid_count if valid_count else 0.0
        ci_low, ci_high = compute_wilson_interval(positive, valid_count)
        qualified = qualifies_case(positive, valid_count, len(group), reliable, all_human)
        gates = []
        if len(group) != 30:
            gates.append("requires_exactly_30_reproduction_samples")
        if positive < 3:
            gates.append("fewer_than_3_positives")
        if not 0.10 <= rate <= 0.80:
            gates.append("rate_outside_0.10_to_0.80")
        if valid_count < 27:
            gates.append("fewer_than_27_valid")
        if not (reliable or all_human):
            gates.append("label_reliability_failed_or_pending")
        if not (reliable or all_human):
            for response in group:
                needs_manual.append(
                    {
                        "run_id": response.run_id,
                        "split": response.split,
                        "pattern_id": response.pattern_id,
                        "behavior_id": response.behavior_id,
                        "sample_index": response.sample_index,
                        "rubric": candidate_by_id[pattern_id]["rubric"],
                        "user_context": response.raw_user_prompt,
                        "response_text": response.response_text,
                        "human_label": human.get((response.split, response.pattern_id, response.sample_index), ""),
                        "ambiguity_flag": "",
                        "notes": "",
                    }
                )
        published = float(candidate_by_id[pattern_id]["match_rate"])
        results.append(
            {
                "pattern_id": pattern_id,
                "behavior_id": candidate_by_id[pattern_id]["behavior_id"],
                "positive_count": positive,
                "valid_count": valid_count,
                "total_count": len(group),
                "empirical_behavior_rate": rate,
                "wilson_95_lower": ci_low,
                "wilson_95_upper": ci_high,
                "published_weirdchat_match_rate": published,
                "descriptive_rate_difference": rate - published,
                "invalid_generation_rate": (len(group) - valid_count) / len(group),
                **metrics,
                "systematic_rubric_error": pattern_id in systematic,
                "all_30_human_labeled": all_human,
                "reliability_pass": reliable,
                "experiment_mode": manifest.experiment_mode,
                "qualifies": qualified,
                "failed_gates": ";".join(gates),
            }
        )
    fields = list(results[0]) if results else ["pattern_id", "qualifies", "failed_gates"]
    _write_csv(run_dir / "reproduction_results.csv", results, fields)
    if needs_manual:
        merged_manual = {(row["split"], row["pattern_id"], str(row["sample_index"])): row for row in audit_rows}
        for row in needs_manual:
            merged_manual[(row["split"], row["pattern_id"], str(row["sample_index"]))] = row
        manual_rows = [merged_manual[key] for key in sorted(merged_manual)]
        _write_csv(
            run_dir / "audit" / "manual_label_required.csv",
            manual_rows,
            list(needs_manual[0]),
        )
    qualifying_ids = select_final_case_ids(results, order)
    status = "COMPLETE" if len(qualifying_ids) == 4 else "INSUFFICIENT_REPRODUCIBLE_CASES"
    frozen_ids = qualifying_ids if status == "COMPLETE" else []
    cases = {
        "status": status,
        "experiment_mode": manifest.experiment_mode,
        "cases": [
            {
                "pattern_id": item,
                "behavior_id": candidate_by_id[item]["behavior_id"],
                "prompt": candidate_by_id[item]["representative_prompt"],
                "rubric": candidate_by_id[item]["rubric"],
                "provenance": {
                    "dataset": manifest.dataset,
                    "dataset_revision": manifest.dataset_revision,
                    "git_commit_sha": manifest.git_commit_sha,
                },
            }
            for item in frozen_ids
        ],
    }
    atomic_write_text(run_dir / "phase_a_cases.yaml", yaml.safe_dump(cases, sort_keys=False, allow_unicode=True))
    _write_reports(run_dir, manifest.experiment_mode, status, results, len(frozen_ids))
    return status


def _write_reports(run_dir: Path, mode: str, status: str, results: list[dict[str, Any]], frozen: int) -> None:
    table = [
        "| Pattern | Positive/valid | Rate (95% Wilson CI) | Audit agreement | Qualifies |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in results:
        agreement = "pending" if row["agreement"] is None else f"{row['agreement']:.3f}"
        table.append(
            f"| {row['pattern_id']} | {row['positive_count']}/{row['valid_count']} | {row['empirical_behavior_rate']:.3f} ({row['wilson_95_lower']:.3f}–{row['wilson_95_upper']:.3f}) | {agreement} | {row['qualifies']} |"
        )
    report = f"""# Phase A report

Status: `{status}`  
Mode: `{mode}`

This run estimates reproduction frequency under the recorded local configuration. It does not establish a causal trigger, shared mechanism, cross-model generality, or exact reproduction when quantization differs. **No trigger-transfer tests were performed.**

Exactly {frozen} cases were frozen. Final estimates use only the untouched reproduction split; smoke outputs are excluded from rates.

## Results

{chr(10).join(table)}

No null-hypothesis test against published rates was performed. Differences are descriptive only.
"""
    atomic_write_text(run_dir / "phase_a_report.md", report)
    bars = []
    for index, row in enumerate(results):
        rate_x = 260 + 600 * row["empirical_behavior_rate"]
        low_x = 260 + 600 * row["wilson_95_lower"]
        high_x = 260 + 600 * row["wilson_95_upper"]
        y = index * 34 + 10
        bars.append(
            f"<text x='0' y='{y + 6}'>{html.escape(row['pattern_id'][:28])}</text>"
            f"<line x1='{low_x}' y1='{y}' x2='{high_x}' y2='{y}' stroke='#4169e1' stroke-width='4'/>"
            f"<circle cx='{rate_x}' cy='{y}' r='6' fill='#172554'/>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Phase A report</title><style>body{font:16px sans-serif;max-width:1000px;margin:auto}pre{white-space:pre-wrap}rect{fill:#4169e1}</style>"
        + f"<h1>Phase A report</h1><p>Status: <code>{status}</code>; mode: <code>{mode}</code>.</p><p>No trigger-transfer tests were performed.</p><svg width='900' height='{max(80, len(results) * 34)}' role='img' aria-label='Empirical reproduction rates'>{''.join(bars)}</svg><pre>{html.escape(report)}</pre>"
    )
    atomic_write_text(run_dir / "phase_a_report.html", document)
    atomic_write_text(run_dir / "reports" / "phase_a_report.md", report)
    atomic_write_text(run_dir / "reports" / "phase_a_report.html", document)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(finalize_run(args.run))


if __name__ == "__main__":
    main()
