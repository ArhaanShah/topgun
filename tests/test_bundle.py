from __future__ import annotations

from datetime import UTC, datetime

import pytest

from phase_a.schemas import GenerationRecord, JudgmentRecord, RunManifest
from phase_a.storage import JSONLStorage, atomic_write_json, atomic_write_text, compute_checksum
from scripts.export_run import export_run
from scripts.verify_bundle import verify_bundle


def _manifest():
    return RunManifest(
        run_id="bundle-run",
        created_at=datetime.now(UTC),
        git_commit_sha="a" * 40,
        git_remote_url="https://example.invalid/repo.git",
        git_dirty=False,
        experiment_mode="mock",
        model_profile="fp8_offload",
        model="mock",
        model_revision="b" * 40,
        tokenizer="mock",
        tokenizer_revision="c" * 40,
        dataset="Transluce/WeirdChat",
        dataset_revision="d" * 40,
        judge_model="judge",
        judge_revision="e" * 40,
        hardware={},
        software={},
        quantization={},
        generation_parameters={},
        hashes={},
        seed_derivation="sha256",
        offline=True,
    )


def _minimal_run(tmp_path, generation_factory, judgment_factory, **judgment_overrides):
    run = tmp_path / "bundle-run"
    (run / "manifests").mkdir(parents=True)
    config_text = "dataset: pinned\n"
    atomic_write_text(run / "config_snapshot" / "phase_a.yaml", config_text)
    manifest = _manifest().model_copy(update={"hashes": {"phase_a.yaml": compute_checksum(config_text)}})
    atomic_write_json(run / "manifests" / "run_manifest.json", manifest)
    atomic_write_json(run / "manifests" / "selection_manifest.json", {"primary_pattern_ids": ["p1"]})
    response = generation_factory(run_id="bundle-run")
    JSONLStorage(run / "responses" / "responses_smoke.jsonl", GenerationRecord).append_record(response)
    JSONLStorage(run / "judgments" / "judgments_smoke.jsonl", JudgmentRecord).append_record(
        judgment_factory(response, **judgment_overrides)
    )
    return run


def test_bundle_creation_and_cpu_verification(tmp_path, generation_factory, judgment_factory):
    run = _minimal_run(tmp_path, generation_factory, judgment_factory)
    bundle, sidecar = export_run(run, tmp_path / "out")
    assert bundle.suffix == ".zst" and sidecar.exists()
    report = verify_bundle(bundle)
    assert report["status"] == "VERIFIED"
    assert report["generation_records"] == 1


def test_bundle_rejects_secrets(tmp_path, generation_factory, judgment_factory):
    run = _minimal_run(tmp_path, generation_factory, judgment_factory)
    (run / "manifests" / "bad.txt").write_text("api_key=do-not-export", encoding="utf-8")
    with pytest.raises(RuntimeError, match="possible secret"):
        export_run(run, tmp_path / "out")


def test_bundle_rejects_judgment_for_different_response(tmp_path, generation_factory, judgment_factory):
    run = _minimal_run(
        tmp_path,
        generation_factory,
        judgment_factory,
        response_checksum="0" * 64,
    )
    bundle, _ = export_run(run, tmp_path / "out")
    with pytest.raises(ValueError, match="judgment response checksum"):
        verify_bundle(bundle)
