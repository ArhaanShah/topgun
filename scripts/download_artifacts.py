#!/usr/bin/env python3
"""Download pinned public artifacts into the external/ignored cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_a.config import REPO_ROOT, load_phase_config, load_profile, load_yaml
from phase_a.environment import guard_local_inference
from phase_a.run import default_cache_dir
from phase_a.storage import atomic_write_json


def hash_small_files(root: Path, limit: int = 64 * 1024 * 1024) -> tuple[dict[str, str], int]:
    hashes, size = {}, 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        current_size = path.stat().st_size
        size += current_size
        if current_size <= limit:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[relative] = digest.hexdigest()
    return hashes, size


def download(profile_name: str, cache_dir: Path, *, dry_run: bool = False) -> dict:
    phase = load_phase_config()
    profile = load_profile(profile_name)
    judge = load_yaml(REPO_ROOT / "configs" / "models" / "local_judge.yaml")
    guard_local_inference(profile)
    artifacts = [("target", profile["model"], profile["revision"]), ("judge", judge["model"], judge["revision"])]
    manifest = {
        "schema_version": 1,
        "artifacts": [],
        "dataset": {"repository": phase["dataset"]["repository"], "revision": phase["dataset"]["revision"]},
    }
    if dry_run:
        manifest["artifacts"] = [
            {"role": role, "repository": repo, "revision": revision} for role, repo, revision in artifacts
        ]
        return manifest
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    for role, repo, revision in artifacts:
        local = cache_dir / "models" / repo.replace("/", "--") / revision
        snapshot_download(repo_id=repo, revision=revision, local_dir=local, token=os.environ.get("HF_TOKEN"))
        hashes, size = hash_small_files(local)
        manifest["artifacts"].append(
            {
                "role": role,
                "repository": repo,
                "revision": revision,
                "cache_path": str(local.resolve()),
                "resolved_size_bytes": size,
                "sha256_files_up_to_64MiB": hashes,
            }
        )
    dataset_root = cache_dir / "datasets" / "Transluce--WeirdChat" / phase["dataset"]["revision"]
    dataset_entries = []
    for subset in ("patterns", "prompts", "rubrics"):
        dataset = load_dataset(
            phase["dataset"]["repository"],
            name=subset,
            revision=phase["dataset"]["revision"],
            split="train",
            cache_dir=str(cache_dir / "huggingface"),
            token=os.environ.get("HF_TOKEN"),
        )
        output = dataset_root / subset
        dataset.save_to_disk(output)
        hashes, size = hash_small_files(output)
        dataset_entries.append(
            {"subset": subset, "cache_path": str(output.resolve()), "resolved_size_bytes": size, "sha256": hashes}
        )
    manifest["dataset"]["subsets"] = dataset_entries
    atomic_write_json(cache_dir / "artifact_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fp8_offload", "awq_4bit"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(args.profile, args.cache_dir, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
