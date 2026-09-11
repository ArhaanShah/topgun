"""CPU-only analysis for the frozen evidence follow-up experiment.

The analysis deliberately joins factors from the trusted schedule to labels by
immutable response ID.  Reviewer supplied factors are never used for contrasts.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "n", "f"}
WINDOWS = ("384", "1024", "full")


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
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    if text == "":
        return None
    raise ValueError(f"invalid Boolean label {value!r}")


class BayesianPosterior:
    """Small standard-library Beta posterior sampler."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, draws: int = 20_000, seed: int = 20260914):
        self.alpha, self.beta, self.draws, self.seed = alpha, beta, draws, seed

    def posterior_interval(self, positives: int, total: int, credible: float = 0.95) -> dict[str, float | int | None]:
        if total < 0 or positives < 0 or positives > total:
            raise ValueError("invalid posterior counts")
        if total == 0:
            return {
                "rate": None,
                "mean": None,
                "median": None,
                "lower": None,
                "upper": None,
                "credible_level": credible,
                "n": 0,
                "positives": 0,
            }
        a, b = self.alpha + positives, self.beta + total - positives
        rng = random.Random(self.seed + positives * 1_000_003 + total)
        draws = sorted(rng.betavariate(a, b) for _ in range(self.draws))
        lo = draws[int((1 - credible) / 2 * (len(draws) - 1))]
        hi = draws[int((1 + credible) / 2 * (len(draws) - 1))]
        return {
            "rate": positives / total,
            "mean": a / (a + b),
            "median": draws[len(draws) // 2],
            "lower": lo,
            "upper": hi,
            "credible_level": credible,
            "n": total,
            "positives": positives,
        }

    def contrast_posterior(
        self, pos_a: int, total_a: int, pos_b: int, total_b: int, credible: float = 0.95
    ) -> dict[str, float | None]:
        if not total_a or not total_b:
            return {"difference": None, "mean_difference": None, "lower": None, "upper": None}
        a1, b1 = self.alpha + pos_a, self.beta + total_a - pos_a
        a2, b2 = self.alpha + pos_b, self.beta + total_b - pos_b
        rng = random.Random(self.seed + pos_a * 97 + pos_b * 193 + total_a * 389 + total_b * 769)
        values = sorted(rng.betavariate(a2, b2) - rng.betavariate(a1, b1) for _ in range(self.draws))
        lo = values[int((1 - credible) / 2 * (len(values) - 1))]
        hi = values[int((1 + credible) / 2 * (len(values) - 1))]
        return {
            "difference": pos_b / total_b - pos_a / total_a,
            "mean_difference": a2 / (a2 + b2) - a1 / (a1 + b1),
            "lower": lo,
            "upper": hi,
        }


class ExperimentAnalyzer:
    def __init__(self, responses_path: Path, labels_path: Path, config: AnalysisConfig | None = None):
        self.responses_path, self.labels_path = Path(responses_path), Path(labels_path)
        self.config = config or AnalysisConfig()
        self.posterior = BayesianPosterior(
            self.config.posterior_prior_alpha,
            self.config.posterior_prior_beta,
            self.config.posterior_draws,
            self.config.seed,
        )
        self.responses: dict[str, dict[str, Any]] = {}
        self.labels: dict[str, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def load_responses(self) -> None:
        rows = self._read(self.responses_path)
        for row in rows:
            rid = row.get("response_id")
            if rid:
                if rid in self.responses:
                    raise ValueError(f"duplicate response ID {rid}")
                self.responses[rid] = row

    def load_labels(self) -> None:
        rows = self._read(self.labels_path)
        for row in rows:
            rid = row.get("response_id")
            if rid:
                if rid in self.labels:
                    raise ValueError(f"duplicate label ID {rid}")
                self.labels[rid] = row

    def _joined(self) -> list[dict[str, Any]]:
        if not self.responses:
            self.load_responses()
        if not self.labels:
            self.load_labels()
        joined = []
        for rid, response in self.responses.items():
            if rid in self.labels:
                joined.append({**response, **self.labels[rid], "response_id": rid})
        self.rows = joined
        return joined

    def _cell(self, rows: list[dict[str, Any]], predicate, window: str) -> dict[str, Any]:
        values = [
            _bool(row.get(f"execution_claim_{window}", row.get(f"label_{window}"))) for row in rows if predicate(row)
        ]
        values = [value for value in values if value is not None]
        positive = sum(values)
        return {
            "positives": positive,
            "total": len(values),
            "missing": len(rows) - len(values),
            "posterior": self.posterior.posterior_interval(positive, len(values)),
        }

    def _contrast(self, rows: list[dict[str, Any]], left, right, window: str) -> dict[str, Any]:
        a, b = self._cell(rows, left, window), self._cell(rows, right, window)
        result = self.posterior.contrast_posterior(a["positives"], a["total"], b["positives"], b["total"])
        return {"left": a, "right": b, **result}

    def get_labeled_subset(self, experiment: str | None = None, window: str = "full") -> dict[str, int]:
        rows = self._joined()
        cell = self._cell(rows, (lambda r: not experiment or r.get("experiment") == experiment), window)
        return {"positives": cell["positives"], "total": cell["total"]}

    def analyze_experiment_effects(self, window: str = "full") -> dict[str, Any]:
        rows = self._joined()
        strata = [(task, wording) for task in ("S", "D", "Z") for wording in (0, 1)]
        a_cells = {
            f"{task}{wording}": self._contrast(
                rows,
                lambda r, t=task, w=wording: (
                    r.get("experiment") == "A"
                    and r.get("task") == t
                    and str(r.get("wording")) == str(w)
                    and str(r.get("evidence")) == "0"
                ),
                lambda r, t=task, w=wording: (
                    r.get("experiment") == "A"
                    and r.get("task") == t
                    and str(r.get("wording")) == str(w)
                    and str(r.get("evidence")) == "1"
                ),
                window,
            )
            for task, wording in strata
        }
        b_by_order = {}
        for order in ("basis-first", "recommendation-first"):
            b_by_order[order] = self._contrast(
                rows,
                lambda r, o=order: r.get("experiment") == "B" and r.get("order") == o and str(r.get("evidence")) == "0",
                lambda r, o=order: r.get("experiment") == "B" and r.get("order") == o and str(r.get("evidence")) == "1",
                window,
            )
        c = self._contrast(
            rows,
            lambda r: r.get("experiment") == "C" and r.get("cue") == "neutral",
            lambda r: r.get("experiment") == "C" and r.get("cue") == "execution-unavailable",
            window,
        )
        return {
            "window": window,
            "A_evidence_effect": {"strata": a_cells},
            "B_evidence_effect": {"by_order": b_by_order},
            "B_order_effect": b_by_order["basis-first"],
            "B_interaction": self.posterior.contrast_posterior(
                b_by_order["recommendation-first"]["right"]["positives"]
                - b_by_order["recommendation-first"]["left"]["positives"],
                max(
                    b_by_order["recommendation-first"]["right"]["total"],
                    b_by_order["recommendation-first"]["left"]["total"],
                    1,
                ),
                b_by_order["basis-first"]["right"]["positives"] - b_by_order["basis-first"]["left"]["positives"],
                max(b_by_order["basis-first"]["right"]["total"], b_by_order["basis-first"]["left"]["total"], 1),
            ),
            "C_cue_effect": c,
        }

    def analyze_task_transfer(self, window: str = "full") -> dict[str, Any]:
        rows = self._joined()
        return {
            f"task_{task}": self._contrast(
                rows,
                lambda r, t=task: r.get("task") == t and str(r.get("evidence")) == "0",
                lambda r, t=task: r.get("task") == t and str(r.get("evidence")) == "1",
                window,
            )
            for task in ("S", "D", "Z")
        }

    def analyze(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._joined()
        report = {
            "config": asdict(self.config),
            "responses_loaded": len(self.responses),
            "labels_loaded": len(self.labels),
            "experiment_effects": {},
            "task_transfer": {},
            "transitions": {},
        }
        for window in WINDOWS:
            report["experiment_effects"][window] = self.analyze_experiment_effects(window)
            report["task_transfer"][window] = self.analyze_task_transfer(window)
        for row in self.rows:
            old = _bool(row.get("execution_claim_1024", row.get("label_1024")))
            new = _bool(row.get("execution_claim_full", row.get("label_full")))
            key = f"{old}->{new}"
            report["transitions"][key] = report["transitions"].get(key, 0) + 1
        (output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def generate_report(self, output_path: Path) -> None:
        report = self.analyze(output_path.parent)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


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
