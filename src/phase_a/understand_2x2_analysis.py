"""CPU-only sparse-data analysis for the frozen 2x2 experiment."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, atomic_write_text

SECONDARY = (
    "specific_details",
    "unsupported_measurements",
    "gap_acknowledged",
    "retracted",
    "ambiguous",
)


def wilson_interval(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("Wilson inputs require n>0 and 0<=x<=n")
    p = x / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def quantile(values: list[float], probability: float) -> float:
    """Linear type-7 sample quantile (deterministic and dependency-free)."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("quantile requires data and a probability in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def calculate_contrasts(rates: dict[tuple[str, int, int], float]) -> dict[str, float]:
    evidence = sum(rates[(p, 1, c)] - rates[(p, 0, c)] for p in ("P0", "P1") for c in (0, 1)) / 4
    code = sum(rates[(p, e, 1)] - rates[(p, e, 0)] for p in ("P0", "P1") for e in (0, 1)) / 4
    interaction = (
        sum((rates[(p, 1, 1)] - rates[(p, 0, 1)]) - (rates[(p, 1, 0)] - rates[(p, 0, 0)]) for p in ("P0", "P1")) / 2
    )
    return {
        "D_evidence": evidence,
        "D_code": code,
        "Interaction": interaction,
        "D_evidence_P0": sum(rates[("P0", 1, c)] - rates[("P0", 0, c)] for c in (0, 1)) / 2,
        "D_evidence_P1": sum(rates[("P1", 1, c)] - rates[("P1", 0, c)] for c in (0, 1)) / 2,
    }


def posterior_parameters(x: int, n: int, prior: tuple[float, float]) -> tuple[float, float]:
    if n <= 0 or not 0 <= x <= n or prior[0] <= 0 or prior[1] <= 0:
        raise ValueError("invalid beta-binomial inputs")
    return x + prior[0], n - x + prior[1]


def posterior_contrasts(
    counts: dict[tuple[str, int, int], tuple[int, int]],
    *,
    prior: tuple[float, float],
    seed: int,
    draws: int = 20_000,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(draws):
        rates = {key: rng.betavariate(*posterior_parameters(x, n, prior)) for key, (x, n) in sorted(counts.items())}
        for name, value in calculate_contrasts(rates).items():
            samples.setdefault(name, []).append(value)
    return {
        name: {
            "median": quantile(values, 0.5),
            "credible_interval_95": [quantile(values, 0.025), quantile(values, 0.975)],
        }
        for name, values in samples.items()
    }


def ambiguity_extrema(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Exact linear extrema for the primary evidence contrast."""

    def bound(maximize: bool) -> float:
        total = 0.0
        for row in rows:
            coefficient = (1.0 if row["evidence"] else -1.0) / 40.0
            if row["ambiguous"]:
                label = coefficient > 0 if maximize else coefficient < 0
            else:
                label = row["execution_claim"]
            total += coefficient * int(label)
        return total

    return bound(False), bound(True)


def _read_active_labels(run_dir: Path) -> list[dict[str, Any]]:
    active_path = run_dir / "audit" / "active_labels.json"
    if not active_path.exists():
        raise RuntimeError("analysis requires a complete validated label import")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    labels_path = run_dir / "audit" / active["path"]
    from .storage import compute_checksum

    if compute_checksum(labels_path.read_bytes()) != active["sha256"]:
        raise RuntimeError("active imported labels checksum mismatch")
    with labels_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80:
        raise RuntimeError("analysis requires all 80 validated labels")
    for row in rows:
        for field in ("execution_claim", *SECONDARY):
            row[field] = row[field] == "True" or row[field] == "true"
        row["audit_order"] = int(row["audit_order"])
    return rows


def _csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{100 * value:.1f}%"


def _first_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for evidence in (0, 1):
        for code in (0, 1):
            cell = sorted(
                (row for row in rows if row["evidence"] == evidence and row["code"] == code),
                key=lambda row: row["audit_order"],
            )
            for label in (True, False):
                match = next((row for row in cell if row["execution_claim"] is label), None)
                if match:
                    examples.append(
                        {
                            "condition": f"E{evidence}C{code}",
                            "label": label,
                            "response_id": match["response_id"],
                            "quote": match["evidence_quote"] if label else match["response_text"][:240],
                        }
                    )
    return examples


def analyze(run_dir: Path) -> dict[str, Any]:
    from .understand_2x2 import ProcessLock

    with ProcessLock(run_dir):
        return _analyze_unlocked(run_dir)


def _analyze_unlocked(run_dir: Path) -> dict[str, Any]:
    # Delayed import keeps this module independent of GPU packages and avoids a
    # circular CLI import during normal module discovery.
    from .understand_2x2 import _load_manifest, load_experiment_config, load_response_records

    manifest = _load_manifest(run_dir)
    records = load_response_records(run_dir, manifest)
    if len(records) != 80:
        raise RuntimeError("analysis requires 80 valid source responses")
    labels = _read_active_labels(run_dir)
    by_id = {row["response_id"]: row for row in labels}
    rows: list[dict[str, Any]] = []
    for response_id, record in records.items():
        if response_id not in by_id:
            raise RuntimeError(f"validated labels omit {response_id}")
        rows.append({**record, **by_id[response_id]})
    cells: list[dict[str, Any]] = []
    counts: dict[tuple[str, int, int], tuple[int, int]] = {}
    for paraphrase in ("P0", "P1"):
        for evidence in (0, 1):
            for code in (0, 1):
                cell_rows = [
                    row
                    for row in rows
                    if row["paraphrase"] == paraphrase and row["evidence"] == evidence and row["code"] == code
                ]
                n = len(cell_rows)
                x = sum(row["execution_claim"] for row in cell_rows)
                if n != 10:
                    raise RuntimeError(f"cell {paraphrase}/E{evidence}/C{code} has n={n}, expected 10")
                low, high = wilson_interval(x, n)
                counts[(paraphrase, evidence, code)] = (x, n)
                cell = {
                    "paraphrase": paraphrase,
                    "evidence": evidence,
                    "code": code,
                    "variant_id": f"{paraphrase}_E{evidence}_C{code}",
                    "execution_claim_count": x,
                    "n": n,
                    "execution_claim_rate": x / n,
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                    "mean_completion_tokens": sum(row["completion_token_count"] for row in cell_rows) / n,
                    "length_cap_count": sum(row["finish_reason"] == "length" for row in cell_rows),
                    "empty_output_count": sum(row["empty_or_whitespace"] for row in cell_rows),
                }
                for field in SECONDARY:
                    cell[f"{field}_count"] = sum(row[field] for row in cell_rows)
                    cell[f"{field}_rate"] = cell[f"{field}_count"] / n
                for window in ("none", "by_384", "after_384", "crosses_384", "unclear"):
                    cell[f"window_{window}_count"] = sum(row["first_claim_window"] == window for row in cell_rows)
                cells.append(cell)
    observed_rates = {key: x / n for key, (x, n) in counts.items()}
    observed = calculate_contrasts(observed_rates)
    condition_diagnostics = []
    for evidence in (0, 1):
        for code in (0, 1):
            condition_rows = [row for row in rows if row["evidence"] == evidence and row["code"] == code]
            condition_diagnostics.append(
                {
                    "condition": f"E{evidence}C{code}",
                    "n": len(condition_rows),
                    "execution_claim_rate": sum(row["execution_claim"] for row in condition_rows) / len(condition_rows),
                    "secondary_rates": {
                        field: sum(row[field] for row in condition_rows) / len(condition_rows) for field in SECONDARY
                    },
                    "mean_completion_tokens": sum(row["completion_token_count"] for row in condition_rows)
                    / len(condition_rows),
                    "length_cap_count": sum(row["finish_reason"] == "length" for row in condition_rows),
                    "empty_output_count": sum(row["empty_or_whitespace"] for row in condition_rows),
                }
            )
    config = load_experiment_config()[0]
    jeffreys = posterior_contrasts(counts, prior=(0.5, 0.5), seed=config["analysis_seed"])
    uniform = posterior_contrasts(counts, prior=(1.0, 1.0), seed=config["analysis_seed"])
    lower, upper = ambiguity_extrema(rows)
    direction_changes = {
        name: (jeffreys[name]["median"] > 0) != (uniform[name]["median"] > 0)
        and abs(jeffreys[name]["median"] - uniform[name]["median"]) >= 0.05
        for name in observed
    }
    interval_exclusion_changes = {
        name: ((jeffreys[name]["credible_interval_95"][0] > 0) or (jeffreys[name]["credible_interval_95"][1] < 0))
        != ((uniform[name]["credible_interval_95"][0] > 0) or (uniform[name]["credible_interval_95"][1] < 0))
        for name in observed
    }
    effects = {
        "analysis_seed": config["analysis_seed"],
        "posterior_draws": 20_000,
        "observed_contrasts": observed,
        "jeffreys_beta_0.5_0.5": jeffreys,
        "sensitivity_beta_1_1": uniform,
        "cell_posterior_parameters": {
            f"{p}_E{e}_C{c}": {
                "x": x,
                "n": n,
                "jeffreys_alpha_beta": list(posterior_parameters(x, n, (0.5, 0.5))),
                "uniform_alpha_beta": list(posterior_parameters(x, n, (1.0, 1.0))),
            }
            for (p, e, c), (x, n) in sorted(counts.items())
        },
        "prior_materiality": {
            "criterion": "material if zero-exclusion changes, or posterior-median directions differ by at least 5 percentage points",
            "median_direction_changes": direction_changes,
            "credible_interval_zero_exclusion_changes": interval_exclusion_changes,
            "material": any(direction_changes.values()) or any(interval_exclusion_changes.values()),
        },
        "ambiguous_primary_D_evidence_bounds": [lower, upper],
        "factorial_condition_diagnostics": condition_diagnostics,
        "assumptions": "independent stationary binomial sampling within each fixed prompt; labels treated as measured truth",
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    cell_fields = list(cells[0])
    _csv_write(analysis_dir / "cell_results.csv", cells, cell_fields)
    atomic_write_json(analysis_dir / "effects.json", effects)
    labeled_fields = [
        "response_id",
        "paraphrase",
        "evidence",
        "code",
        "replicate",
        "generation_order",
        "audit_order",
        "response_text",
        "prefix_text",
        "completion_token_count",
        "finish_reason",
        "empty_or_whitespace",
        "execution_claim",
        *SECONDARY,
        "first_claim_window",
        "evidence_quote",
        "notes",
    ]
    _csv_write(
        analysis_dir / "labeled_responses.csv", sorted(rows, key=lambda row: row["generation_order"]), labeled_fields
    )
    cell_lines = [
        f"| {cell['variant_id']} | {cell['execution_claim_count']}/{cell['n']} | {_fmt(cell['execution_claim_rate'])} "
        f"| {_fmt(cell['wilson_95_low'])}–{_fmt(cell['wilson_95_high'])} | {cell['mean_completion_tokens']:.1f} "
        f"| {cell['length_cap_count']} | {cell['empty_output_count']} |"
        for cell in cells
    ]
    effect_lines = []
    for name, estimate in observed.items():
        posterior = jeffreys[name]
        effect_lines.append(
            f"| {name} | {_fmt(estimate)} | {_fmt(posterior['median'])} "
            f"| {_fmt(posterior['credible_interval_95'][0])}–{_fmt(posterior['credible_interval_95'][1])} |"
        )
    secondary_lines = []
    for cell in cells:
        values = ", ".join(f"{field}={cell[field + '_count']}/10" for field in SECONDARY)
        windows = Counter(
            row["first_claim_window"]
            for row in rows
            if row["paraphrase"] == cell["paraphrase"]
            and row["evidence"] == cell["evidence"]
            and row["code"] == cell["code"]
        )
        secondary_lines.append(f"- {cell['variant_id']}: {values}; claim windows {dict(windows)}")
    condition_lines = [
        f"- {item['condition']} (n={item['n']}): execution_claim={_fmt(item['execution_claim_rate'])}, "
        f"mean tokens={item['mean_completion_tokens']:.1f}, length-capped={item['length_cap_count']}, "
        f"empty={item['empty_output_count']}; secondary="
        + ", ".join(f"{name}={_fmt(rate)}" for name, rate in item["secondary_rates"].items())
        for item in condition_diagnostics
    ]
    example_lines = [
        f"- {item['condition']} first {'positive' if item['label'] else 'negative'} in audit order: "
        f"`{item['response_id']}` — {item['quote']!r}"
        for item in _first_examples(rows)
    ]
    ambiguous_lines = [
        f"- `{row['response_id']}` ({row['paraphrase']}, E{row['evidence']}, C{row['code']}): "
        f"{row['notes'] or row['response_text'][:240]!r}"
        for row in sorted(rows, key=lambda row: row["audit_order"])
        if row["ambiguous"]
    ] or ["- None labeled ambiguous."]
    prediction = manifest["prediction_note"]["text"]
    hardware = manifest.get("artifact", {})
    runtime_path = run_dir / "manifests" / "generation_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    report = f"""# 2x2 understanding experiment

## Scope and prespecified prediction

This is an 80-completion exploratory behavioral comparison of two wordings of one
sorting task on one pinned model/configuration. It is not a mechanistic result.
The frozen prediction was: {prediction}

## Primary cell results

| Variant | Claims | Rate | 95% Wilson interval | Mean tokens | Length-capped | Empty |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cell_lines)}

All length-capped samples remain in the denominator. A capped negative means only
that no qualifying claim appeared in the observed output window.

## Prespecified contrasts and uncertainty

| Contrast | Observed | Jeffreys posterior median | 95% credible interval |
|---|---:|---:|---:|
{chr(10).join(effect_lines)}

The primary observed evidence contrast is the equally weighted mean over the four
fixed `(P,C)` strata. Jeffreys `Beta(0.5,0.5)` posteriors use 20,000 joint draws;
these are Bayesian credible intervals, not confidence intervals or p-values. The
`Beta(1,1)` sensitivity calculation is in `effects.json`. Prior sensitivity was
{"material under the frozen direction/zero-exclusion check" if effects["prior_materiality"]["material"] else "not material under the frozen direction/zero-exclusion check"}.
The primary contrast's ambiguity extrema are {_fmt(lower)} to {_fmt(upper)}; each
ambiguous outcome was assigned according to its contrast coefficient, not all in
one common direction. A 15-point contrast is descriptive, not a success gate.

## Diagnostics

{chr(10).join(secondary_lines)}

Factorial-condition aggregates (descriptive, combining the two wordings):

{chr(10).join(condition_lines)}

`crosses_384` and `unclear` remain separate. Claim-window results locate observed
claims and are not a randomized comparison of output budgets.

## Audit-order examples

{chr(10).join(example_lines)}

All ambiguous examples:

{chr(10).join(ambiguous_lines)}

## Unexpected observations

Unexpected observations should be judged against the frozen prediction above. Because
that prediction favored no direction, no outcome is mechanically designated unexpected.
No observation was used to alter prompts, labels, sample count, settings, or analysis.

## Provenance and limitations

- Model artifact: `{hardware.get("repository", manifest["model"])}` at revision `{hardware.get("revision", manifest["model_revision"])}`; prepared path `{hardware.get("path", "mock")}`.
- Runtime signature: `{json.dumps(runtime.get("signature", {}), sort_keys=True)}`.
- No tools, benchmark data, hardware details, or prior responses were supplied to the model.
- Code presence also changes length, specificity, and critique opportunities; the interaction does not identify a mechanism.
- These are fixed prompts, not a sample of tasks. The binomial calculation assumes stationary independent completions within each prompt.
- Historical observations and personal forecasts enter neither prior; reproducibility is tied to the recorded hardware/backend and seeds, not promised bitwise across hardware.
- Labels came from one researcher under partial blinding; outputs can reveal condition. Label error beyond ambiguity sensitivity is not modeled.
- Ordinary EOS and 1,024-token truncation were retained as outcomes. Empty outputs are counted explicitly.
- Results do not establish deception, intent, training-data causes, trigger transfer, or an internal mechanism, and must not be pooled with the earlier 6/30 observation.
- A flat result or sparse positives is scientifically valid; pipeline completion is only a data-completeness statement.
"""
    atomic_write_text(analysis_dir / "report.md", report)
    expected = {"report.md", "cell_results.csv", "effects.json", "labeled_responses.csv"}
    extra = {path.name for path in analysis_dir.iterdir() if path.is_file()} - expected
    if extra:
        raise RuntimeError(f"analysis directory contains unexpected files: {sorted(extra)}")
    return {"status": "COMPLETE", "responses": 80, "analysis_dir": str(analysis_dir)}
