#!/usr/bin/env python3
"""CPU-only integrity, schema, count, uniqueness, and secret verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_a.schemas import GenerationRecord, JudgmentRecord, RunManifest
from phase_a.storage import canonical_json, compute_checksum

SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|authorization|bearer)\s*[:=]\s*(?!<(?:set|unset|redacted)>)[^\s,}]+"
)


def _sidecar(bundle: Path) -> Path:
    return bundle.with_name(bundle.name + ".sha256")


def _validate_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError(f"unsafe archive member: {name}")


def _load_wrapped(path: Path, schema: type) -> list:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        wrapped = json.loads(line)
        data = wrapped["data"]
        data_text = data if isinstance(data, str) else canonical_json(data)
        if compute_checksum(data_text) != wrapped["checksum"]:
            raise ValueError(f"record checksum failed: {path}:{line_number}")
        records.append(schema.model_validate_json(data_text))
    return records


def verify_bundle(bundle_path: str | Path) -> dict:
    bundle = Path(bundle_path).resolve()
    checksum_path = _sidecar(bundle)
    if not bundle.exists() or not checksum_path.exists():
        raise FileNotFoundError("download both the bundle and its .sha256 sidecar")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("bundle SHA-256 mismatch")
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("zstandard is required to verify .tar.zst bundles") from exc
    with tempfile.TemporaryDirectory(prefix="phase-a-verify-") as temp:
        temp_dir = Path(temp)
        tar_path = temp_dir / "bundle.tar"
        with bundle.open("rb") as source, tar_path.open("wb") as destination:
            zstandard.ZstdDecompressor().copy_stream(source, destination)
        with tarfile.open(tar_path, "r") as archive:
            for member in archive.getmembers():
                _validate_member(member.name)
                if member.issym() or member.islnk():
                    raise ValueError("links are not allowed in a run bundle")
            archive.extractall(temp_dir, filter="data")
        roots = [path for path in temp_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ValueError("bundle must contain exactly one run directory")
        run = roots[0]
        manifest_path = run / "manifests" / "run_manifest.json"
        selection_path = run / "manifests" / "selection_manifest.json"
        if not manifest_path.exists() or not selection_path.exists():
            raise ValueError("Git/configuration and selection manifests are required")
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        for name, expected_hash in manifest.hashes.items():
            if name in {"tokenizer", "chat_template"}:
                continue
            path = (
                run / "prompts" / "rendered_prompts.jsonl"
                if name == "rendered_prompts"
                else run / "config_snapshot" / name
            )
            if not path.exists() or compute_checksum(path.read_text(encoding="utf-8")) != expected_hash:
                raise ValueError(f"manifest hash failed: {name}")
        generations, judgments = [], []
        for split in ("smoke", "reproduction"):
            response_path = run / "responses" / f"responses_{split}.jsonl"
            judgment_path = run / "judgments" / f"judgments_{split}.jsonl"
            if response_path.exists():
                generations.extend(_load_wrapped(response_path, GenerationRecord))
            if judgment_path.exists():
                judgments.extend(_load_wrapped(judgment_path, JudgmentRecord))
        generation_ids = [(r.run_id, r.split, r.pattern_id, r.sample_index) for r in generations]
        judgment_ids = [(r.run_id, r.split, r.pattern_id, r.sample_index) for r in judgments]
        if len(generation_ids) != len(set(generation_ids)) or len(judgment_ids) != len(set(judgment_ids)):
            raise ValueError("duplicate sample IDs")
        if set(generation_ids) != set(judgment_ids):
            raise ValueError("every target response must have exactly one judgment")
        responses_by_id = {(r.run_id, r.split, r.pattern_id, r.sample_index): r.response_text for r in generations}
        for judgment in judgments:
            identity = (judgment.run_id, judgment.split, judgment.pattern_id, judgment.sample_index)
            if judgment.response_checksum != compute_checksum(responses_by_id[identity]):
                raise ValueError(f"judgment response checksum failed: {identity}")
        if manifest.experiment_mode != "mock":
            required = {
                "phase_a_cases.yaml",
                "reproduction_results.csv",
                "responses.jsonl",
                "judgments.jsonl",
                "human_audit.csv",
                "phase_a_report.md",
                "phase_a_report.html",
            }
            missing = sorted(name for name in required if not (run / name).exists())
            if missing:
                raise ValueError("production bundle is missing final deliverables: " + ", ".join(missing))
            smoke_expected = len(selection["primary_pattern_ids"]) * 10
            advanced_path = run / "selections" / "advanced_candidates.csv"
            with advanced_path.open(newline="", encoding="utf-8") as handle:
                reproduction_expected = len(list(csv.DictReader(handle))) * 30
            actual_smoke = sum(r.split == "smoke" for r in generations)
            actual_reproduction = sum(r.split == "reproduction" for r in generations)
            if (actual_smoke, actual_reproduction) != (smoke_expected, reproduction_expected):
                raise ValueError(f"unexpected sample counts: {(actual_smoke, actual_reproduction)}")
        for path in run.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".json",
                ".jsonl",
                ".yaml",
                ".yml",
                ".txt",
                ".md",
                ".csv",
                ".html",
            }:
                if SECRET_PATTERN.search(path.read_text(encoding="utf-8", errors="replace")):
                    raise ValueError(f"possible secret in bundle: {path.relative_to(run)}")
        return {
            "status": "VERIFIED",
            "run_id": manifest.run_id,
            "generation_records": len(generations),
            "judgment_records": len(judgments),
            "sha256": actual,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle), indent=2))


if __name__ == "__main__":
    main()
