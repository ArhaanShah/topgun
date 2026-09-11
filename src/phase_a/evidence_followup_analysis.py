"""CPU-only, schedule-trusted analysis for the evidence follow-up experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "n", "f"}
WINDOWS = ("384", "1024", "full")
OUTCOMES = ("execution_claim", "unsupported_measurements", "union")
LABEL_BASES = ("execution_claim", "unsupported_measurements", "ambiguous", "retracted", "evidence_quote")


@dataclass
class AnalysisConfig:
    posterior_prior_alpha: float = 0.5
    posterior_prior_beta: float = 0.5
    posterior_draws: int = 20_000
    sensitivity_prior_alpha: float = 1.0
    sensitivity_prior_beta: float = 1.0
    credible_interval: float = 0.95
    seed: int = 20260914


def _bool(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"invalid Boolean label {value!r}")


class BayesianPosterior:
    """Independent Beta cell posteriors and linear-combination summaries."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, draws: int = 20_000, seed: int = 20260914):
        self.alpha, self.beta, self.draws, self.seed = alpha, beta, draws, seed

    def draws_for(self, positives: int, total: int, key: str = "") -> list[float]:
        if total < 0 or positives < 0 or positives > total:
            raise ValueError("invalid posterior counts")
        if total == 0:
            return []
        salt = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        rng = random.Random(self.seed + salt)
        return [rng.betavariate(self.alpha + positives, self.beta + total - positives) for _ in range(self.draws)]

    @staticmethod
    def summarize(values: list[float], credible: float = 0.95) -> dict[str, float | None]:
        if not values:
            return {"mean_difference": None, "lower": None, "upper": None}
        ordered = sorted(values)
        return {
            "mean_difference": sum(values) / len(values),
            "lower": ordered[int((1 - credible) / 2 * (len(values) - 1))],
            "upper": ordered[int((1 + credible) / 2 * (len(values) - 1))],
        }

    def posterior_interval(self, positives: int, total: int, credible: float = 0.95) -> dict[str, Any]:
        draws = self.draws_for(positives, total, f"interval:{positives}:{total}")
        summary = self.summarize(draws, credible)
        return {
            "rate": positives / total if total else None,
            "mean": summary["mean_difference"],
            "median": sorted(draws)[len(draws) // 2] if draws else None,
            "lower": summary["lower"], "upper": summary["upper"], "credible_level": credible,
            "n": total, "positives": positives,
        }

    def contrast_posterior(
        self, pos_a: int, total_a: int, pos_b: int, total_b: int, credible: float = 0.95
    ) -> dict[str, float | None]:
        left = self.draws_for(pos_a, total_a, f"contrast:left:{pos_a}:{total_a}")
        right = self.draws_for(pos_b, total_b, f"contrast:right:{pos_b}:{total_b}")
        values = [b - a for a, b in zip(left, right, strict=True)] if left and right else []
        return {
            "difference": pos_b / total_b - pos_a / total_a if total_a and total_b else None,
            **self.summarize(values, credible),
        }


class ExperimentAnalyzer:
    def __init__(
        self, responses_path: Path, labels_path: Path, config: AnalysisConfig | None = None,
        schedule_path: Path | None = None,
    ):
        self.responses_path, self.labels_path = Path(responses_path), Path(labels_path)
        self.schedule_path = schedule_path or self.responses_path.parent / "schedule.jsonl"
        self.config = config or AnalysisConfig()
        self.posterior = BayesianPosterior(
            self.config.posterior_prior_alpha, self.config.posterior_prior_beta,
            self.config.posterior_draws, self.config.seed,
        )
        self.responses: dict[str, dict[str, Any]] = {}
        self.labels: dict[str, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self._draws: dict[str, list[float]] = {}
        self._cell_draw_cache: dict[tuple[str, str, tuple[str, ...]], list[float]] = {}

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def load_responses(self) -> None:
        for row in self._read(self.responses_path):
            rid = row.get("response_id")
            if not rid or rid in self.responses:
                raise ValueError(f"missing or duplicate response ID {rid}")
            self.responses[rid] = row

    def load_labels(self) -> None:
        for row in self._read(self.labels_path):
            rid = row.get("response_id")
            if not rid or rid in self.labels:
                raise ValueError(f"missing or duplicate label ID {rid}")
            self.labels[rid] = row

    def _joined(self) -> list[dict[str, Any]]:
        if not self.responses:
            self.load_responses()
        if not self.labels:
            self.load_labels()
        schedule_rows = self._read(self.schedule_path)
        schedule = {row["response_id"]: row for row in schedule_rows}
        if schedule_rows and len(schedule) != len(schedule_rows):
            raise ValueError("trusted schedule has duplicate identities")
        allowed = {f"{base}_{window}" for base in LABEL_BASES for window in WINDOWS}
        joined = []
        for rid, response in self.responses.items():
            if rid not in self.labels:
                continue
            if schedule_rows and rid not in schedule:
                raise ValueError(f"response is absent from trusted schedule: {rid}")
            factors = schedule.get(rid, response)
            trusted = {key: factors.get(key) for key in
                       ("experiment", "task", "wording", "evidence", "order", "cue", "replicate")}
            labels = {key: value for key, value in self.labels[rid].items() if key in allowed}
            joined.append({**response, **trusted, **labels, "response_id": rid})
        self.rows = joined
        return joined

    @staticmethod
    def _value(row: dict[str, Any], outcome: str, window: str) -> bool | None:
        if outcome == "union":
            a = _bool(row.get(f"execution_claim_{window}"))
            b = _bool(row.get(f"unsupported_measurements_{window}"))
            return None if a is None or b is None else a or b
        return _bool(row.get(f"{outcome}_{window}"))

    def _cell(
        self, rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], window: str,
        outcome: str = "execution_claim", key: str = "",
    ) -> dict[str, Any]:
        filtered = [row for row in rows if predicate(row)]
        values = [self._value(row, outcome, window) for row in filtered]
        present = [value for value in values if value is not None]
        positives = sum(present)
        identity = tuple(sorted(row["response_id"] for row, value in zip(filtered, values, strict=True) if value is not None))
        cache_key = (outcome, window, identity)
        if cache_key not in self._cell_draw_cache:
            self._cell_draw_cache[cache_key] = self.posterior.draws_for(
                positives, len(present), f"{outcome}:{window}:{'|'.join(identity)}"
            )
        draws = self._cell_draw_cache[cache_key]
        self._draws[key] = draws
        summary = self.posterior.summarize(draws, self.config.credible_interval)
        return {
            "positives": positives, "total": len(present), "cell_rows": len(filtered),
            "missing": len(filtered) - len(present),
            "posterior": {
                "rate": positives / len(present) if present else None,
                "mean": summary["mean_difference"],
                "median": sorted(draws)[len(draws) // 2] if draws else None,
                "lower": summary["lower"], "upper": summary["upper"],
                "credible_level": self.config.credible_interval, "n": len(present), "positives": positives,
            },
        }

    def _contrast(self, rows, left, right, window: str, outcome: str = "execution_claim", key: str = ""):
        a = self._cell(rows, left, window, outcome, key + ":left")
        b = self._cell(rows, right, window, outcome, key + ":right")
        da, db = self._draws[key + ":left"], self._draws[key + ":right"]
        draws = [y - x for x, y in zip(da, db, strict=True)] if da and db else []
        return {
            "left": a, "right": b, "_draws": draws,
            "difference": b["positives"] / b["total"] - a["positives"] / a["total"]
            if a["total"] and b["total"] else None,
            **self.posterior.summarize(draws, self.config.credible_interval),
        }

    @staticmethod
    def _public(contrast: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in contrast.items() if key != "_draws"}

    def get_labeled_subset(self, experiment: str | None = None, window: str = "full") -> dict[str, int]:
        cell = self._cell(self._joined(), lambda r: not experiment or r.get("experiment") == experiment, window)
        return {"positives": cell["positives"], "total": cell["total"]}

    def analyze_experiment_effects(self, window: str = "full", outcome: str = "execution_claim") -> dict[str, Any]:
        rows = self._joined()
        strata = [(task, wording) for task in ("S", "D", "Z") for wording in (0, 1)]

        def pred(exp: str, task: str, wording: int, **factors: Any):
            return lambda r: (
                r.get("experiment") == exp and r.get("task") == task
                and str(r.get("wording")) == str(wording)
                and all(str(r.get(k)) == str(v) for k, v in factors.items())
            )

        a: dict[str, Any] = {}
        a_draws: list[list[float]] = []
        b_evidence: dict[str, Any] = {}
        b_draws: dict[str, list[list[float]]] = {"basis-first": [], "recommendation-first": []}
        b_order: dict[str, Any] = {}
        b_order_draws: list[list[float]] = []
        interactions: list[list[float]] = []
        for task, wording in strata:
            skey = f"{task}{wording}"
            ac = self._contrast(rows, pred("A", task, wording, evidence=0), pred("A", task, wording, evidence=1),
                                window, outcome, f"A:{skey}")
            a[skey] = self._public(ac)
            a_draws.append(ac["_draws"])
            bcells = {}
            raw_b = {}
            for order in ("basis-first", "recommendation-first"):
                raw_b[order] = self._contrast(
                    rows, pred("B", task, wording, evidence=0, order=order),
                    pred("B", task, wording, evidence=1, order=order), window, outcome, f"B:{skey}:{order}",
                )
                bcells[order] = self._public(raw_b[order])
                b_draws[order].append(raw_b[order]["_draws"])
            b_evidence[skey] = bcells
            order_cells = [self._contrast(
                rows, pred("B", task, wording, evidence=e, order="basis-first"),
                pred("B", task, wording, evidence=e, order="recommendation-first"),
                window, outcome, f"BO:{skey}:{e}",
            ) for e in (0, 1)]
            order_draws = [(x + y) / 2 for x, y in zip(order_cells[0]["_draws"], order_cells[1]["_draws"], strict=True)]
            b_order[skey] = {"by_evidence": [self._public(x) for x in order_cells],
                             **self.posterior.summarize(order_draws, self.config.credible_interval)}
            b_order_draws.append(order_draws)
            interactions.append([x - y for x, y in zip(
                raw_b["basis-first"]["_draws"], raw_b["recommendation-first"]["_draws"], strict=True
            )])
        c = {}
        c_draws: list[list[float]] = []
        for task, wording in strata:
            skey = f"{task}{wording}"
            raw_c = self._contrast(
                rows, pred("C", task, wording, cue="neutral"),
                pred("C", task, wording, cue="execution-unavailable"), window, outcome, f"C:{skey}",
            )
            c[skey] = self._public(raw_c)
            c_draws.append(raw_c["_draws"])
        def equal_average(draw_sets: list[list[float]]) -> list[float]:
            return [sum(draw) / 6 for draw in zip(*draw_sets, strict=True)] if all(draw_sets) else []
        aggregate = [sum(draw) / 6 for draw in zip(*interactions, strict=True)] if all(interactions) else []
        return {
            "window": window, "outcome": outcome,
            "A_evidence_effect": {"strata": a, "equal_stratum_aggregate": self.posterior.summarize(
                equal_average(a_draws), self.config.credible_interval)},
            "B_evidence_effect": {"strata": b_evidence, "equal_stratum_aggregate_by_order": {
                order: self.posterior.summarize(equal_average(draws), self.config.credible_interval)
                for order, draws in b_draws.items()}},
            "B_order_effect": {"strata": b_order, "equal_stratum_aggregate": self.posterior.summarize(
                equal_average(b_order_draws), self.config.credible_interval)},
            "B_interaction": {
                **self.posterior.summarize(aggregate, self.config.credible_interval),
                "formula": "(E1-E0)_basis-first - (E1-E0)_recommendation-first",
            },
            "C_cue_effect": {"strata": c, "equal_stratum_aggregate": self.posterior.summarize(
                equal_average(c_draws), self.config.credible_interval)},
        }

    def analyze_task_transfer(self, window: str = "full", outcome: str = "execution_claim") -> dict[str, Any]:
        rows = self._joined()
        result: dict[str, Any] = {"A": {}, "B": {}, "C": {}}
        for task in ("S", "D", "Z"):
            result["A"][f"task_{task}"] = self._public(self._contrast(
                rows, lambda r, t=task: r.get("experiment") == "A" and r.get("task") == t and str(r.get("evidence")) == "0",
                lambda r, t=task: r.get("experiment") == "A" and r.get("task") == t and str(r.get("evidence")) == "1",
                window, outcome, f"TA:{task}",
            ))
            result["B"][f"task_{task}"] = self._public(self._contrast(
                rows, lambda r, t=task: r.get("experiment") == "B" and r.get("task") == t and str(r.get("evidence")) == "0",
                lambda r, t=task: r.get("experiment") == "B" and r.get("task") == t and str(r.get("evidence")) == "1",
                window, outcome, f"TB:{task}",
            ))
            result["C"][f"task_{task}"] = self._public(self._contrast(
                rows, lambda r, t=task: r.get("experiment") == "C" and r.get("task") == t and r.get("cue") == "neutral",
                lambda r, t=task: r.get("experiment") == "C" and r.get("task") == t and r.get("cue") == "execution-unavailable",
                window, outcome, f"TC:{task}",
            ))
        return result

    def analyze(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._joined()
        report: dict[str, Any] = {
            "config": asdict(self.config), "responses_loaded": len(self.responses),
            "labels_loaded": len(self.labels), "outcomes": {}, "transitions": {}, "capped_outcome_bounds": {},
            "observed_status": {},
        }
        schedule_count = len(self._read(self.schedule_path))
        report["technical_results"] = {
            "planned": schedule_count or len(self.responses),
            "collected": len(self.responses),
            "missing_unresolved": max(0, schedule_count - len(self.responses)) if schedule_count else 0,
        }
        for outcome in OUTCOMES:
            effects, transfer = {}, {}
            for window in WINDOWS:
                effects[window] = self.analyze_experiment_effects(window, outcome)
                transfer[window] = self.analyze_task_transfer(window, outcome)
            report["outcomes"][outcome] = {"experiment_effects": effects, "task_transfer": transfer}
            transitions: dict[str, int] = {}
            for row in rows:
                key = f"{self._value(row, outcome, '1024')}->{self._value(row, outcome, 'full')}"
                transitions[key] = transitions.get(key, 0) + 1
            report["transitions"][outcome] = transitions
            report["observed_status"][outcome] = {
                "ever_asserted_full": sum(self._value(r, outcome, "full") is True for r in rows),
                "observed_end_unretracted": sum(
                    self._value(r, outcome, "full") is True and _bool(r.get("retracted_full")) is False for r in rows
                ),
            }
            capped = [r for r in rows if r.get("finish_reason") == "length" or r.get("completion_token_count") == 4096]
            positive = sum(self._value(r, outcome, "full") is True for r in capped)
            negative = sum(self._value(r, outcome, "full") is False for r in capped)
            ambiguous = sum(_bool(r.get("ambiguous_full")) is True for r in capped)
            positive_unambiguous = sum(
                self._value(r, outcome, "full") is True and _bool(r.get("ambiguous_full")) is False for r in capped
            )
            positive_or_ambiguous = sum(
                self._value(r, outcome, "full") is True or _bool(r.get("ambiguous_full")) is True for r in capped
            )
            report["capped_outcome_bounds"][outcome] = {
                "capped": len(capped), "observed_positive": positive, "observed_negative": negative,
                "ambiguous": ambiguous,
                "ambiguity_extremes": [positive_unambiguous, positive_or_ambiguous],
                "eventual_occurrence_bounds": [positive, len(capped)],
                "note": "A capped positive establishes occurrence, not whether later unseen text would retract it.",
            }
        report["experiment_effects"] = report["outcomes"]["execution_claim"]["experiment_effects"]
        report["task_transfer"] = report["outcomes"]["execution_claim"]["task_transfer"]
        (output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def generate_report(self, output_path: Path) -> None:
        output_path.write_text(json.dumps(self.analyze(output_path.parent), indent=2), encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ExperimentAnalyzer(args.responses, args.labels).analyze(args.output)


if __name__ == "__main__":
    main()
