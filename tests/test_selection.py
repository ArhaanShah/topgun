from __future__ import annotations

import polars as pl

from phase_a.dataset import normalize_patterns
from phase_a.select_candidates import apply_filters, select_candidates


def _frames(count=15):
    patterns, rubrics = [], []
    for index in range(count):
        behavior = f"behavior-{index if index != 14 else 0}"
        patterns.append(
            {
                "pattern_id": f"p{index:02d}",
                "behavior_id": behavior,
                "subject_model": "qwen/qwen3.6-27b",
                "checkpoint": "x",
                "representative_user_text": "natural prompt",
                "metrics": {"match_rate": 0.2},
                "elo": {"prompt_naturalness": {"elo": 1300.0}, "harmfulness": {"elo": 1000.0}},
                "n_transcripts": 64,
            }
        )
        rubrics.append(
            {
                "behavior_id": behavior,
                "transcript_rubric": {"rubric_id": f"r{index:02d}", "text": "rubric"},
            }
        )
    return pl.DataFrame(patterns), pl.DataFrame(rubrics)


def test_real_nested_schema_and_deterministic_selection():
    patterns, rubrics = _frames()
    normalized = normalize_patterns(patterns, rubrics, token_counter=lambda text: len(text.split()))
    eligible = apply_filters(normalized, {})
    primary1, reserve1 = select_candidates(eligible)
    primary2, reserve2 = select_candidates(eligible)
    assert primary1.equals(primary2) and reserve1.equals(reserve2)
    assert len(primary1) == 8 and len(reserve1) == 4
    combined = pl.concat([primary1, reserve1])
    assert combined["behavior_id"].n_unique() == len(combined)
    assert combined["rubric"].to_list() == ["rubric"] * len(combined)


def test_safety_and_boundary_filters():
    patterns, rubrics = _frames(3)
    normalized = normalize_patterns(patterns, rubrics, token_counter=lambda text: 301)
    assert apply_filters(normalized, {}).is_empty()
    normalized = normalize_patterns(patterns, rubrics, token_counter=lambda text: 2)
    assert len(apply_filters(normalized, {"behavior-1": "excluded"})) == 2
