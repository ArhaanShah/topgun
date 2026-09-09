#!/usr/bin/env python3
"""Create a compact, allowlisted .tar.zst run bundle and checksum."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_a.storage import atomic_write_text

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
            if path.suffix in {".zst", ".sha256"}:
                continue
            yield path, relative


def _scan_secrets(path: Path) -> None:
    if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".csv", ".html"}:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if SECRET_PATTERN.search(text):
        raise RuntimeError(f"possible secret in exportable file: {path}")


def export_run(run_dir: str | Path, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not (run_dir / "manifests" / "run_manifest.json").exists():
        raise RuntimeError("run_manifest.json is required")
    manifest = __import__("json").loads((run_dir / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
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
        compressor = zstandard.ZstdCompressor(level=10)
        with tar_path.open("rb") as source, temporary.open("wb") as destination:
            compressor.copy_stream(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, bundle)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    checksum = bundle.with_name(bundle.name + ".sha256")
    atomic_write_text(checksum, f"{digest}  {bundle.name}\n")
    return bundle, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    bundle, checksum = export_run(args.run_dir, args.output_dir)
    print(bundle)
    print(checksum)


if __name__ == "__main__":
    main()
