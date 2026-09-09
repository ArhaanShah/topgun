from __future__ import annotations

from datetime import UTC, datetime

import pytest

from phase_a.schemas import GenerationRecord, JudgmentRecord
from phase_a.storage import compute_checksum


@pytest.fixture
def generation_factory():
    def make(**overrides):
        values = {
            "run_id": "test-run",
            "commit_sha": "a" * 40,
            "experiment_mode": "mock",
            "model_profile": "fp8_offload",
            "split": "smoke",
            "pattern_id": "p1",
            "behavior_id": "b1",
            "sample_index": 0,
            "seed": 42,
            "raw_user_prompt": "hello",
            "rendered_prompt_hash": "b" * 64,
            "response_text": "alpha response",
            "finish_reason": "stop",
            "prompt_token_count": 1,
            "completion_token_count": 2,
            "model": "mock/model",
            "model_revision": "c" * 40,
            "tokenizer": "mock/tokenizer",
            "tokenizer_revision": "d" * 40,
            "repository_revision": "c" * 40,
            "chat_template_hash": "e" * 64,
            "quantization_metadata": {},
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 384,
            "reasoning_setting": False,
            "generation_timestamp": datetime.now(UTC),
            "generation_duration": 0.1,
            "validity_status": True,
            "error_type": None,
            "error_message": None,
        }
        values.update(overrides)
        return GenerationRecord(**values)

    return make


@pytest.fixture
def judgment_factory():
    def make(response=None, **overrides):
        response_text = response.response_text if response else "alpha response"
        values = {
            "label": True,
            "evidence": ["alpha"],
            "reason": "The response contains alpha.",
            "invalid": False,
            "run_id": response.run_id if response else "test-run",
            "split": response.split if response else "smoke",
            "pattern_id": response.pattern_id if response else "p1",
            "behavior_id": response.behavior_id if response else "b1",
            "sample_index": response.sample_index if response else 0,
            "response_checksum": compute_checksum(response_text),
            "judge_model": "judge",
            "judge_revision": "f" * 40,
            "judged_at": datetime.now(UTC),
            "parse_attempts": 1,
            "raw_judge_output": "{}",
        }
        values.update(overrides)
        return JudgmentRecord(**values)

    return make
