"""Runtime policy, environment reporting, and Git provenance."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .storage import atomic_write_json

REMOTE_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
}


def guard_local_inference(config: dict[str, Any]) -> None:
    """Reject any selected hosted backend or non-loopback inference URL."""
    backend = str(config.get("backend", "vllm")).lower()
    if backend not in {"vllm", "mock"}:
        raise RuntimeError(f"remote inference backend is forbidden: {backend}")
    url = str(config.get("base_url", "")).strip().lower()
    if url and not any(url.startswith(prefix) for prefix in ("http://127.0.0.1", "http://localhost", "unix://")):
        raise RuntimeError(f"remote inference URL is forbidden: {url}")
    configured = sorted(key for key in REMOTE_KEYS if os.environ.get(key))
    if configured and backend != "mock":
        raise RuntimeError(
            "remote-provider credentials are set; unset them before local inference: " + ", ".join(configured)
        )


def enable_offline_mode() -> None:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )


def git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    remotes = run("remote")
    return {
        "commit_sha": run("rev-parse", "HEAD"),
        "remote_url": run("remote", "get-url", "origin") if remotes else "",
        "dirty": bool(run("status", "--porcelain")),
    }


def require_clean_git(repo: Path, allow_dirty: bool = False) -> dict[str, Any]:
    info = git_info(repo)
    if info["dirty"] and not allow_dirty:
        raise RuntimeError("working tree is dirty; production runs require an immutable clean commit")
    return info


def package_versions(names: tuple[str, ...] = ("torch", "vllm", "transformers", "datasets")) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_report(path: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "system_ram_bytes": None,
        "gpu": [],
        "cuda_runtime": None,
        "torch_cuda_available": False,
    }
    try:
        import psutil

        report["system_ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        import torch

        report["cuda_runtime"] = torch.version.cuda
        report["torch_cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 4:
                report["gpu"].append(
                    {
                        "name": parts[0],
                        "memory_total_mib": int(parts[1]),
                        "driver_version": parts[2],
                        "compute_capability": parts[3],
                    }
                )
    if path:
        atomic_write_json(path, report)
    return report


def redacted_environment() -> dict[str, str]:
    safe = {"PHASE_A_CACHE_DIR", "PHASE_A_RUNS_DIR", "HF_HOME", "HF_HUB_OFFLINE"}
    output = {key: os.environ[key] for key in safe if key in os.environ}
    output["HF_TOKEN"] = "<set>" if os.environ.get("HF_TOKEN") else "<unset>"
    return output
