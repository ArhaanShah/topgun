from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from phase_a.config import load_profile
from phase_a.inference import Completion, VLLMBackend
from phase_a.understand_2x2 import (
    BOOL_FIELDS,
    ProcessLock,
    SimpleTokenizer,
    build_schedule,
    export_audit,
    import_audit,
    load_experiment_config,
    load_response_records,
    prepare_experiment,
    render_raw_variants,
    run_experiment,
)
from phase_a.understand_2x2_analysis import (
    ambiguity_extrema,
    calculate_contrasts,
    posterior_contrasts,
    posterior_parameters,
    quantile,
    wilson_interval,
)


def _prepared(tmp_path: Path, run_id: str = "test-2x2") -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    prepare_experiment(run_id, cache, runs, mock=True)
    return cache, runs / run_id


class CountingBackend:
    def __init__(self, fail_at: int | None = None, empty: bool = False):
        self.calls = 0
        self.fail_at = fail_at
        self.empty = empty

    def generate(self, prompts, seeds, parameters):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic technical failure")
        text = "canary noun output" if self.calls == 1 else ("" if self.empty else f"sample {seeds[0]}")
        token_ids = list(range(len(text.split())))
        return [
            Completion(text, "length" if self.empty else "stop", len(prompts[0].split()), len(token_ids), token_ids)
        ]


def test_exact_variants_and_balanced_independent_schedule():
    config, prompts, _ = load_experiment_config()
    variants = render_raw_variants(config, prompts)
    assert len(variants) == 8
    assert all(prompts["baseline_instruction"] in row["raw_user_text"] for row in variants)
    assert all((prompts["evidence_instruction"] in row["raw_user_text"]) == bool(row["evidence"]) for row in variants)
    assert all((prompts["code"] in row["raw_user_text"]) == bool(row["code"]) for row in variants)
    schedule = build_schedule(config)
    assert (
        len(schedule) == len({row["response_id"] for row in schedule}) == len({row["seed"] for row in schedule}) == 80
    )
    assert all(len({row["variant_id"] for row in schedule if row["block"] == block}) == 8 for block in range(10))
    mutated = {**config, "unrelated_timestamp": "2099-01-01", "run_id": "different"}
    assert build_schedule(mutated) == schedule


def test_cpp_quicksort_matches_std_sort(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ is not installed")
    _, prompts, _ = load_experiment_config()
    code = prompts["code"].removeprefix("```cpp\n").removesuffix("\n```")
    source = tmp_path / "quicksort_test.cpp"
    source.write_text(
        code
        + r"""
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>
void check(std::vector<double> a) {
    auto expected = a;
    std::sort(expected.begin(), expected.end());
    quicksort(a.data(), a.size());
    assert(a == expected);
}
int main() {
    check({}); check({1.0}); check({2,2,1,1,3,3});
    check({1,2,3,4,5}); check({5,4,3,2,1});
    std::mt19937_64 rng(123); std::uniform_real_distribution<double> d(0.0, 1.0);
    std::vector<double> random(10000); for (auto& value : random) value = d(rng); check(random);
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "quicksort_test"
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O2", str(source), "-o", str(binary)], capture_output=True, text=True, check=False
    )
    legacy_compiler = False
    if "unrecognized command line option" in compiled.stderr:
        legacy_compiler = True
        compiled = subprocess.run(
            [compiler, "-std=c++0x", "-O2", str(source), "-o", str(binary)],
            capture_output=True,
            text=True,
            check=False,
        )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(binary)], check=True)
    if legacy_compiler:
        pytest.skip("algorithm validated, but installed g++ cannot validate C++20 mode")


def test_completion_is_backwards_compatible():
    completion = Completion("text", "stop", 1, 1)
    assert completion.token_ids is None


def test_vllm_forwards_only_explicit_engine_settings(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("vllm")
    fake.LLM = FakeLLM
    fake.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake)
    profile = load_profile("qwen3_6_27b_awq_a100")
    VLLMBackend(profile)
    assert captured["generation_config"] == "vllm"
    assert captured["seed"] == 0
    old = load_profile("awq_4bit")
    captured.clear()
    VLLMBackend(old)
    assert "generation_config" not in captured and "seed" not in captured


def test_interrupted_resume_preserves_existing_bytes_and_one_backend(tmp_path):
    cache, run_dir = _prepared(tmp_path)
    first = CountingBackend(fail_at=5)
    with pytest.raises(RuntimeError, match="TECHNICAL_ERROR_RETRYABLE"):
        run_experiment(run_dir, cache, resume=True, mock=True, backend=first, tokenizer=SimpleTokenizer())
    saved = {path.name: path.read_bytes() for path in (run_dir / "responses").glob("*.json")}
    assert len(saved) == 3  # first call was the separate canary
    second = CountingBackend()
    result = run_experiment(run_dir, cache, resume=True, mock=True, backend=second, tokenizer=SimpleTokenizer())
    assert result["written"] == 77
    assert all((run_dir / "responses" / name).read_bytes() == content for name, content in saved.items())
    assert second.calls == 78  # one canary per engine launch, then 77 pending requests


def test_complete_rerun_does_not_touch_backend(tmp_path):
    cache, run_dir = _prepared(tmp_path)
    backend = CountingBackend()
    run_experiment(run_dir, cache, resume=True, mock=True, backend=backend, tokenizer=SimpleTokenizer())
    calls = backend.calls
    result = run_experiment(run_dir, cache, resume=True, mock=True, backend=backend, tokenizer=SimpleTokenizer())
    assert result == {"existing": 80, "pending": 0, "written": 0, "status": "COMPLETE"}
    assert backend.calls == calls


def test_empty_length_capped_outputs_are_valid_outcomes(tmp_path):
    cache, run_dir = _prepared(tmp_path)
    run_experiment(
        run_dir, cache, resume=True, mock=True, backend=CountingBackend(empty=True), tokenizer=SimpleTokenizer()
    )
    records = load_response_records(run_dir)
    assert len(records) == 80
    assert all(record["empty_or_whitespace"] and record["finish_reason"] == "length" for record in records.values())


def test_corruption_and_concurrent_writer_fail_loudly(tmp_path):
    cache, run_dir = _prepared(tmp_path)
    run_experiment(run_dir, cache, resume=True, mock=True, backend=CountingBackend(), tokenizer=SimpleTokenizer())
    response = next((run_dir / "responses").glob("*.json"))
    original = response.read_text(encoding="utf-8")
    response.write_text(original.replace("sample", "tampered", 1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_response_records(run_dir)
    response.write_text(original, encoding="utf-8")
    with ProcessLock(run_dir), pytest.raises(RuntimeError, match="another writer"):
        with ProcessLock(run_dir):
            pass


def test_audit_is_blind_escaped_and_multiline_round_trips(tmp_path):
    cache, run_dir = _prepared(tmp_path)
    backend = CountingBackend()
    run_experiment(run_dir, cache, resume=True, mock=True, backend=backend, tokenizer=SimpleTokenizer())
    first = next((run_dir / "responses").glob("*.json"))
    record = json.loads(first.read_text(encoding="utf-8"))
    # Rebuild this record through the same checksum convention with hostile multiline text.
    record["response_text"] = "<script>alert(1)</script>\nsecond line"
    record["prefix_text"] = record["response_text"]
    record["record_hash"] = __import__("phase_a.understand_2x2", fromlist=["_hash_value"])._hash_value(
        {key: value for key, value in record.items() if key != "record_hash"}
    )
    first.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    export_audit(run_dir)
    html_text = (run_dir / "audit" / "audit_review.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in html_text and "<script>alert" not in html_text
    with (run_dir / "audit" / "human_labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["response_text"] == record["response_text"] for row in rows)
    forbidden = {"paraphrase", "evidence", "code", "seed", "finish_reason", "variant_id"}
    assert forbidden.isdisjoint(rows[0])


def test_label_validation_and_versioned_import(tmp_path):
    cache, run_dir = _prepared(tmp_path)
    run_experiment(run_dir, cache, resume=True, mock=True, backend=CountingBackend(), tokenizer=SimpleTokenizer())
    export_audit(run_dir)
    source = run_dir / "audit" / "human_labels.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in BOOL_FIELDS:
            row[field] = "false"
        row["first_claim_window"] = "none"
    labels = tmp_path / "completed.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = import_audit(run_dir, labels)
    second = import_audit(run_dir, labels)
    assert first.name == "imported_labels_v1.csv" and second.name == "imported_labels_v2.csv"
    rows[0]["execution_claim"] = "true"
    rows[0]["first_claim_window"] = "by_384"
    rows[0]["evidence_quote"] = "not an exact quote"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="exact response substring"):
        import_audit(run_dir, labels)


def test_statistics_signs_endpoints_and_determinism():
    rates = {(p, e, c): float(e) for p in ("P0", "P1") for e in (0, 1) for c in (0, 1)}
    contrasts = calculate_contrasts(rates)
    assert contrasts["D_evidence"] == 1 and contrasts["D_code"] == contrasts["Interaction"] == 0
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0) and high == pytest.approx(0.2775328)
    counts = {key: (0, 10) for key in rates}
    assert posterior_parameters(0, 10, (0.5, 0.5)) == (0.5, 10.5)
    one = posterior_contrasts(counts, prior=(0.5, 0.5), seed=20260911, draws=500)
    two = posterior_contrasts(counts, prior=(0.5, 0.5), seed=20260911, draws=500)
    assert one == two
    assert one["D_evidence"]["credible_interval_95"][0] < one["D_evidence"]["credible_interval_95"][1]
    counts = {key: (10, 10) for key in rates}
    all_one = posterior_contrasts(counts, prior=(0.5, 0.5), seed=1, draws=500)
    assert all_one["D_code"]["credible_interval_95"][0] < all_one["D_code"]["credible_interval_95"][1]
    assert quantile([0, 10], 0.25) == 2.5


def test_ambiguity_extrema_matches_exhaustive_tiny_case():
    rows = [
        {"evidence": 1, "ambiguous": True, "execution_claim": False},
        {"evidence": 0, "ambiguous": True, "execution_claim": True},
        {"evidence": 1, "ambiguous": False, "execution_claim": True},
    ]
    assert ambiguity_extrema(rows) == pytest.approx((0.0, 0.05))
