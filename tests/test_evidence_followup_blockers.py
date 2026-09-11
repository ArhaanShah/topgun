from __future__ import annotations

import csv
import json
import sys
import types

import pytest

from phase_a import evidence_followup as ef
from phase_a.config import load_profile
from phase_a.evidence_followup_analysis import AnalysisConfig, ExperimentAnalyzer
from phase_a.inference import Completion, VLLMBackend


class DistinctTokenizer:
    name_or_path = "distinct-production-tokenizer"
    chat_template = "DISTINCT::{content}::GENERATE"

    def __len__(self):
        return 999

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(token) for token in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert kwargs["enable_thinking"] is False
        return f"DISTINCT::{messages[0]['content']}::GENERATE"


class ProductionStubBackend:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = []

    def generate(self, prompts, seeds, parameters):
        self.prompts.extend(prompts)
        count = parameters["max_tokens"] if parameters["max_tokens"] == 4096 else 4
        reason = "length" if count == 4096 else "stop"
        ids = [65 + index % 2 for index in range(count)]
        return [Completion("original backend text", reason, len(self.tokenizer.encode(prompts[0])), count, ids)]


class InterruptingBackend:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def generate(self, prompts, seeds, parameters):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        return [Completion(f"sample {seeds[0]}", "stop", len(prompts[0].split()), 2, [1, 2])]


def test_vllm_stress_path_forces_length_and_ordinary_path_allows_eos(monkeypatch):
    sampling_calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            sampling_calls.append(kwargs)
            self.ignore_eos = kwargs.get("ignore_eos", False)

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling, use_tqdm):
            count = 4096 if sampling.ignore_eos else 1
            reason = "length" if sampling.ignore_eos else "stop"
            candidate = types.SimpleNamespace(text="a", finish_reason=reason, token_ids=list(range(count)))
            request = types.SimpleNamespace(outputs=[candidate], prompt_token_ids=[1, 2])
            return [request]

    fake = types.ModuleType("vllm")
    fake.LLM = FakeLLM
    fake.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)
    backend = VLLMBackend(load_profile("qwen3_6_27b_awq_a100_followup"))

    ordinary = backend.generate(["prompt"], [1], {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096})[0]
    stress = backend.generate_technical_stress(["prompt"], [1])[0]

    assert ordinary.finish_reason == "stop"
    assert ordinary.completion_tokens == 1
    assert stress.finish_reason == "length"
    assert stress.completion_tokens == 4096
    assert sampling_calls[0] == {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096, "seed": 1}
    assert sampling_calls[1] == {"max_tokens": 4096, "min_tokens": 4096, "ignore_eos": True, "seed": 1}


def test_production_uses_and_reverifies_real_tokenizer(monkeypatch, tmp_path):
    tokenizer = DistinctTokenizer()
    cache, run = tmp_path / "cache", tmp_path / "run"
    cache.mkdir()
    monkeypatch.setattr(ef, "git_info", lambda _: {"commit_sha": "abc", "remote_url": "x", "dirty": False})
    monkeypatch.setattr(ef, "verify_target", lambda *a, **k: {
        "repository": "model", "revision": "revision", "resolved_size_bytes": 1, "small_metadata_sha256": {}
    })
    monkeypatch.setattr(ef, "_load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(ef, "gpu_preflight", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(ef, "_runtime_signature", lambda mock: {"mock": mock})

    class ForbiddenSimple:
        name_or_path = ef.SimpleTokenizer.name_or_path

        def __init__(self):
            raise AssertionError("SimpleTokenizer was constructed in production")

    monkeypatch.setattr(ef, "SimpleTokenizer", ForbiddenSimple)
    ef.prepare_experiment("run", cache, run, mock=False)
    schedule = ef._schedule(run)
    assert schedule[0]["prompt"]["rendered_prompt"].startswith("DISTINCT::")
    assert schedule[0]["prompt"]["rendered_token_count"] == len(tokenizer.encode(schedule[0]["prompt"]["rendered_prompt"]))
    backend = ProductionStubBackend(tokenizer)
    ef.run_experiment(run, cache, resume=True, mock=False, backend=backend, tokenizer=tokenizer, limit=1)
    assert backend.prompts[2] == schedule[0]["prompt"]["rendered_prompt"]
    record = ef._records(run)[schedule[0]["response_id"]]
    assert record["response_text"] == "original backend text"
    assert record["prefix_384_token_ids"] == record["output_token_ids"][:384]
    assert record["decoded_384"] == tokenizer.decode(record["output_token_ids"][:384])


@pytest.mark.parametrize(
    "completion, message",
    [
        (Completion("x", "stop", 1, 1, None), "omitted exact output"),
        (Completion("x", "stop", 1, 2, [1]), "accounting mismatch"),
        (Completion("x", "mystery", 1, 1, [1]), "unknown"),
        (Completion("x", "length", 1, 4097, list(range(4097))), "exceeded"),
    ],
)
def test_completion_validation_rejects_malformed_results(completion, message):
    with pytest.raises(RuntimeError, match=message):
        ef._validate_completion(completion, 1, 4096)


def test_interrupted_followup_resume_never_overwrites_committed_records(tmp_path):
    cache, run = tmp_path / "cache", tmp_path / "run"
    cache.mkdir()
    ef.prepare_experiment("run", cache, run, mock=True)
    first = InterruptingBackend(fail_at=4)
    with pytest.raises(RuntimeError, match="TECHNICAL_ERROR_RETRYABLE"):
        ef.run_experiment(run, cache, resume=True, mock=True, backend=first, tokenizer=ef.SimpleTokenizer())
    before = {path.name: path.read_bytes() for path in (run / "responses").glob("*.json")}
    assert len(before) == 3
    ef.run_experiment(run, cache, resume=True, mock=True, backend=InterruptingBackend(), tokenizer=ef.SimpleTokenizer())
    after = ef._records(run)
    assert len(after) == 144
    assert all((run / "responses" / name).read_bytes() == content for name, content in before.items())
    assert len({record["response_id"] for record in after.values()}) == 144


def test_four_cell_interaction_positive_and_task_transfer_is_separate(tmp_path):
    responses, labels = [], []
    rid = 0
    for task in ("S", "D", "Z"):
        for wording in (0, 1):
            for experiment, cells in (
                ("A", [(0, None, None, False), (1, None, None, True)]),
                ("B", [
                    (0, "basis-first", None, False), (1, "basis-first", None, True),
                    (0, "recommendation-first", None, True), (1, "recommendation-first", None, False),
                ]),
                ("C", [(1, None, "neutral", False), (1, None, "execution-unavailable", True)]),
            ):
                for evidence, order, cue, positive in cells:
                    for replicate in range(3):
                        rid += 1
                        response_id = f"r{rid}"
                        responses.append({"response_id": response_id, "experiment": experiment, "task": task,
                                          "wording": wording, "evidence": evidence, "order": order, "cue": cue,
                                          "replicate": replicate, "finish_reason": "stop", "completion_token_count": 10})
                        label = {"response_id": response_id}
                        for window in ("384", "1024", "full"):
                            label[f"execution_claim_{window}"] = str(positive).lower()
                            label[f"unsupported_measurements_{window}"] = str(not positive).lower()
                            label[f"ambiguous_{window}"] = "false"
                            label[f"retracted_{window}"] = "false"
                            label[f"evidence_quote_{window}"] = ""
                        labels.append(label)
    response_path, label_path = tmp_path / "responses.jsonl", tmp_path / "labels.csv"
    response_path.write_text("".join(json.dumps(row) + "\n" for row in responses), encoding="utf-8")
    with label_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    analyzer = ExperimentAnalyzer(response_path, label_path, AnalysisConfig(posterior_draws=1000))
    effects = analyzer.analyze_experiment_effects("full")
    assert effects["B_interaction"]["mean_difference"] > 0
    assert effects["B_interaction"]["formula"] == "(E1-E0)_basis-first - (E1-E0)_recommendation-first"
    assert abs(effects["B_order_effect"]["strata"]["S0"]["mean_difference"]) < 0.2
    transfer = analyzer.analyze_task_transfer("full")
    assert set(transfer) == {"A", "B", "C"}
    assert transfer["C"]["task_S"]["difference"] == 1.0
