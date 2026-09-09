from __future__ import annotations

import csv

import yaml

from phase_a.audit import create_audit_package
from phase_a.finalize import finalize_run
from phase_a.generate import run_generation
from phase_a.judge import run_judge
from phase_a.prepare import build_mock_frames, prepare_run
from phase_a.run import initialize_manifest, resolve_run_dir
from phase_a.smoke_test import run_health_check
from scripts.export_run import export_run
from scripts.verify_bundle import verify_bundle


def test_tiny_mock_pipeline_through_verified_export(tmp_path, monkeypatch):
    run = resolve_run_dir(tmp_path / "runs", "tiny", create=True)
    cache = tmp_path / "cache"
    initialize_manifest(run, "fp8_offload", mock=True, allow_dirty=True, offline=True)
    prepare_run(run, "fp8_offload", cache, mock_frames=build_mock_frames())
    assert run_health_check(run, "fp8_offload", mock=True)["status"] == "PASS"
    run_generation(run, "smoke", profile_name="fp8_offload", mock=True, limit=2)
    run_judge(run, "smoke", mock=True, limit=2)

    selected_path = run / "selections" / "selected_candidates.csv"
    with selected_path.open(newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    advanced_path = run / "selections" / "advanced_candidates.csv"
    with advanced_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)

    from phase_a import generate

    real_config = generate.load_phase_config()
    tiny_config = {**real_config, "sampling": {**real_config["sampling"], "reproduction_per_candidate": 2}}
    monkeypatch.setattr(generate, "load_phase_config", lambda: tiny_config)
    run_generation(run, "reproduction", profile_name="fp8_offload", mock=True)
    run_judge(run, "reproduction", mock=True)
    create_audit_package(run)
    assert finalize_run(run) == "INSUFFICIENT_REPRODUCIBLE_CASES"
    assert yaml.safe_load((run / "phase_a_cases.yaml").read_text(encoding="utf-8"))["cases"] == []
    bundle, _ = export_run(run, tmp_path / "out")
    assert verify_bundle(bundle)["generation_records"] == 4
