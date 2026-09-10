from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from phase_a.finalize import select_final_case_ids
from phase_a.schemas import RunManifest


def _values():
    return {
        "run_id": "run",
        "created_at": datetime.now(UTC),
        "git_commit_sha": "a" * 40,
        "git_remote_url": "git@example.invalid:x/y",
        "git_dirty": False,
        "experiment_mode": "exact_configuration_attempt",
        "model_profile": "fp8_offload",
        "model": "model",
        "model_revision": "b" * 40,
        "tokenizer": "tokenizer",
        "tokenizer_revision": "c" * 40,
        "dataset": "dataset",
        "dataset_revision": "d" * 40,
        "judge_model": "judge",
        "judge_revision": "e" * 40,
        "hardware": {},
        "software": {},
        "quantization": {},
        "generation_parameters": {},
        "resolved_configuration": {},
        "profile_configuration": {},
        "paths": {},
        "disk": {},
        "hashes": {},
        "seed_derivation": "sha256",
        "offline": True,
    }


def test_required_manifest_fields_and_clean_production():
    values = _values()
    assert RunManifest(**values).offline
    with pytest.raises(ValidationError):
        RunManifest(**(values | {"git_dirty": True}))
    values.pop("model_revision")
    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_final_cases_follow_original_order_not_rate():
    results = [
        {"pattern_id": "p3", "qualifies": True, "rate": 0.7},
        {"pattern_id": "p1", "qualifies": True, "rate": 0.1},
        {"pattern_id": "p2", "qualifies": True, "rate": 0.8},
    ]
    assert select_final_case_ids(results, ["p1", "p2", "p3"], 2) == ["p1", "p2"]
