"""Download and offline verification for pinned public artifacts."""

from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, load_phase_config, load_profile, load_yaml
from .environment import enable_offline_mode, guard_local_inference
from .storage import atomic_write_json


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


def download(profile_name: str, cache_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    phase = load_phase_config()
    profile = load_profile(profile_name)
    judge = load_yaml(REPO_ROOT / "configs" / "models" / "local_judge.yaml")
    guard_local_inference(profile)
    artifacts = [("target", profile["model"], profile["revision"]), ("judge", judge["model"], judge["revision"])]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "cache_dir": str(cache_dir.resolve()),
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
    entries = []
    for subset in phase["dataset"]["subsets"]:
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
        entries.append(
            {"subset": subset, "cache_path": str(output.resolve()), "resolved_size_bytes": size, "sha256": hashes}
        )
    manifest["dataset"]["subsets"] = entries
    atomic_write_json(cache_dir / "artifact_manifest.json", manifest)
    return manifest


def verify_offline(profile_name: str, cache_dir: Path) -> dict[str, Any]:
    profile = load_profile(profile_name)
    judge = load_yaml(REPO_ROOT / "configs" / "models" / "local_judge.yaml")
    phase = load_phase_config()
    enable_offline_mode()
    original = socket.socket.connect

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted during offline verification")

    socket.socket.connect = blocked
    checked = []
    try:
        from transformers import AutoConfig, AutoTokenizer

        for role, item in (("target", profile), ("judge", judge)):
            local = cache_dir / "models" / item["model"].replace("/", "--") / item["revision"]
            AutoConfig.from_pretrained(local, local_files_only=True, trust_remote_code=False)
            AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
            checked.append({"role": role, "path": str(local)})
        from datasets import load_from_disk

        dataset_root = cache_dir / "datasets" / "Transluce--WeirdChat" / phase["dataset"]["revision"]
        for subset in phase["dataset"]["subsets"]:
            load_from_disk(dataset_root / subset)
            checked.append({"role": f"dataset:{subset}", "path": str(dataset_root / subset)})
    finally:
        socket.socket.connect = original
    return {"offline": True, "network_calls": 0, "checked": checked}
