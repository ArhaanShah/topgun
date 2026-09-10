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
    primary1 = select_candidates(eligible)
    primary2 = select_candidates(eligible)
    assert primary1.equals(primary2)
    assert len(primary1) == 8
    assert primary1["behavior_id"].n_unique() == len(primary1)
    assert primary1["rubric"].to_list() == ["rubric"] * len(primary1)


def test_safety_and_boundary_filters():
    patterns, rubrics = _frames(3)
    normalized = normalize_patterns(patterns, rubrics, token_counter=lambda text: 301)
    assert apply_filters(normalized, {}).is_empty()
    normalized = normalize_patterns(patterns, rubrics, token_counter=lambda text: 2)
    assert len(apply_filters(normalized, {"behavior-1": "excluded"})) == 2


def test_fixture_configuration_changes_filter_behavior():
    patterns, rubrics = _frames(3)
    normalized = normalize_patterns(patterns, rubrics, token_counter=lambda text: 2)
    from phase_a.config import load_phase_config

    selection = load_phase_config()["selection"]
    assert len(apply_filters(normalized, {}, selection)) == 3
    changed = {**selection, "match_rate_min": 0.3}
    assert apply_filters(normalized, {}, changed).is_empty()
