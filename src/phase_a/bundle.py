"""Allowlisted run export and CPU-only bundle verification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .schemas import GenerationRecord, JudgmentRecord, RunManifest
from .storage import atomic_write_text, canonical_json, compute_checksum

ALLOWED_DIRS = {
    "config_snapshot",
    "manifests",
    "selections",
    "prompts",
    "responses",
    "judgments",
    "audit",
    "reports",
    "checkpoints",
}
ALLOWED_ROOT = {
    "phase_a_cases.yaml",
    "reproduction_results.csv",
    "responses.jsonl",
    "judgments.jsonl",
    "human_audit.csv",
    "phase_a_report.md",
    "phase_a_report.html",
}
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|authorization|bearer)\s*[:=]\s*(?!<(?:set|unset|redacted)>)[^\s,}]+"
)


def _files(run_dir: Path):
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir)
        if relative.parts[0] in ALLOWED_DIRS or relative.as_posix() in ALLOWED_ROOT:
            if path.suffix not in {".zst", ".sha256"}:
                yield path, relative


def _scan_secrets(path: Path) -> None:
    if path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".csv", ".html"}:
        if SECRET_PATTERN.search(path.read_text(encoding="utf-8", errors="replace")):
            raise RuntimeError(f"possible secret in exportable file: {path}")


def export_run(run_dir: str | Path, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    if not run_dir.is_dir() or not manifest_path.exists():
        raise RuntimeError("a run directory with run_manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_mode") != "mock":
        missing = sorted(name for name in ALLOWED_ROOT if not (run_dir / name).exists())
        if missing:
            raise RuntimeError("finalize the production run before export; missing: " + ", ".join(missing))
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("zstandard is required to export .tar.zst bundles") from exc
    output = Path(output_dir).resolve() if output_dir else run_dir.parent
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"phase_a_{run_dir.name}.tar.zst"
    with tempfile.TemporaryDirectory(prefix="phase-a-export-") as temp:
        tar_path = Path(temp) / "run.tar"
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for path, relative in _files(run_dir):
                _scan_secrets(path)
                info = archive.gettarinfo(str(path), arcname=f"{run_dir.name}/{relative.as_posix()}")
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
        temporary = bundle.with_suffix(bundle.suffix + ".tmp")
        with tar_path.open("rb") as source, temporary.open("wb") as destination:
            zstandard.ZstdCompressor(level=10).copy_stream(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, bundle)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    checksum = bundle.with_name(bundle.name + ".sha256")
    atomic_write_text(checksum, f"{digest}  {bundle.name}\n")
    return bundle, checksum


def _load_wrapped(path: Path, schema: type) -> list[Any]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        wrapped = json.loads(line)
        data = wrapped["data"]
        text = data if isinstance(data, str) else canonical_json(data)
        if compute_checksum(text) != wrapped["checksum"]:
            raise ValueError(f"record checksum failed: {path}:{line_number}")
        records.append(schema.model_validate_json(text))
    return records


def verify_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    sidecar = bundle.with_name(bundle.name + ".sha256")
    if not bundle.exists() or not sidecar.exists():
        raise FileNotFoundError("download both the bundle and its .sha256 sidecar")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("bundle SHA-256 mismatch")
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("zstandard is required to verify .tar.zst bundles") from exc
    with tempfile.TemporaryDirectory(prefix="phase-a-verify-") as temp:
        root = Path(temp)
        tar_path = root / "bundle.tar"
        with bundle.open("rb") as source, tar_path.open("wb") as destination:
            zstandard.ZstdDecompressor().copy_stream(source, destination)
        with tarfile.open(tar_path, "r") as archive:
            for member in archive.getmembers():
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or len(pure.parts) < 2 or member.issym() or member.islnk():
                    raise ValueError(f"unsafe archive member: {member.name}")
            archive.extractall(root, filter="data")
        roots = [path for path in root.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ValueError("bundle must contain exactly one run directory")
        run = roots[0]
        manifest = RunManifest.model_validate_json(
            (run / "manifests" / "run_manifest.json").read_text(encoding="utf-8")
        )
        selection = json.loads((run / "manifests" / "selection_manifest.json").read_text(encoding="utf-8"))
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
        responses = {
            identity: record.response_text for identity, record in zip(generation_ids, generations, strict=True)
        }
        for identity, judgment in zip(judgment_ids, judgments, strict=True):
            if judgment.response_checksum != compute_checksum(responses[identity]):
                raise ValueError(f"judgment response checksum failed: {identity}")
        if manifest.experiment_mode != "mock":
            smoke_expected = (
                len(selection["primary_pattern_ids"])
                * manifest.resolved_configuration["sampling"]["smoke_per_candidate"]
            )
            with (run / "selections" / "advanced_candidates.csv").open(newline="", encoding="utf-8") as handle:
                advanced = len(list(csv.DictReader(handle)))
            reproduction_expected = advanced * manifest.resolved_configuration["sampling"]["reproduction_per_candidate"]
            actual_counts = (
                sum(r.split == "smoke" for r in generations),
                sum(r.split == "reproduction" for r in generations),
            )
            if actual_counts != (smoke_expected, reproduction_expected):
                raise ValueError(f"unexpected sample counts: {actual_counts}")
        return {
            "status": "VERIFIED",
            "run_id": manifest.run_id,
            "generation_records": len(generations),
            "judgment_records": len(judgments),
            "sha256": actual,
        }
