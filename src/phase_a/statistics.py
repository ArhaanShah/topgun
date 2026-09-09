"""Phase A descriptive statistics and audit reliability."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist


def compute_wilson_interval(positive_count: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if positive_count < 0 or n < 0 or positive_count > n:
        raise ValueError("require 0 <= positive_count <= n")
    if n == 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = positive_count / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def reliability_metrics(pairs: Iterable[tuple[bool, bool]]) -> dict[str, float | int | None]:
    pairs = list(pairs)
    tp = sum(auto and human for auto, human in pairs)
    tn = sum(not auto and not human for auto, human in pairs)
    fp = sum(auto and not human for auto, human in pairs)
    fn = sum(not auto and human for auto, human in pairs)
    n = len(pairs)
    agreement = (tp + tn) / n if n else None
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    if n:
        p_auto = (tp + fp) / n
        p_human = (tp + fn) / n
        expected = p_auto * p_human + (1 - p_auto) * (1 - p_human)
        kappa = (agreement - expected) / (1 - expected) if expected < 1 else 1.0
    else:
        kappa = None
    return {
        "audit_n": n,
        "agreement": agreement,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "cohens_kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def passes_reliability(metrics: dict[str, float | int | None]) -> bool:
    return (
        metrics["agreement"] is not None
        and float(metrics["agreement"]) >= 0.90
        and metrics["sensitivity"] is not None
        and float(metrics["sensitivity"]) >= 0.80
        and metrics["specificity"] is not None
        and float(metrics["specificity"]) >= 0.80
    )


def qualifies_case(
    positive: int, valid: int, total: int, reliable: bool, all_human_labeled: bool, safety_ok: bool = True
) -> bool:
    rate = positive / valid if valid else 0.0
    return (
        positive >= 3
        and 0.10 <= rate <= 0.80
        and valid >= 27
        and total == 30
        and (reliable or all_human_labeled)
        and safety_ok
    )
