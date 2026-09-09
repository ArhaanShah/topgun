import pytest

from phase_a.config import REPO_ROOT
from phase_a.environment import guard_local_inference, redacted_environment
from phase_a.select_candidates import load_exclusions


def test_exact_safety_ids_are_machine_readable():
    exclusions = load_exclusions(REPO_ROOT / "configs" / "safety_exclusions.yaml")
    assert "direct-kill-yourself" in exclusions
    assert "recommends-drunk-driving" in exclusions
    assert all(exclusions.values())


def test_remote_inference_is_rejected(monkeypatch):
    with pytest.raises(RuntimeError):
        guard_local_inference({"backend": "openai"})
    with pytest.raises(RuntimeError):
        guard_local_inference({"backend": "vllm", "base_url": "https://example.com"})
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    with pytest.raises(RuntimeError):
        guard_local_inference({"backend": "vllm"})
    assert "super-secret" not in str(redacted_environment())
