"""L4 resource and software preflight."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from packaging.version import Version

from .config import load_profile
from .environment import environment_report, guard_local_inference
from .storage import atomic_write_json


def run_preflight(profile_name: str, cache_dir: Path, output: Path, *, strict: bool = True) -> dict[str, Any]:
    profile = load_profile(profile_name)
    guard_local_inference(profile)
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_report()
    disk = shutil.disk_usage(cache_dir)
    gpu_lines = environment["gpu"]
    l4 = any("L4" in item["name"] for item in gpu_lines)
    vllm_version = environment["packages"].get("vllm", "not-installed")
    vllm_installed = vllm_version != "not-installed"
    vllm_checkpoint_support = vllm_installed and Version(vllm_version) >= Version("0.19.0")
    l4_kernel_capability = l4 and all(float(item["compute_capability"]) >= 8.9 for item in gpu_lines)
    required_disk = int(profile["estimated_download_bytes"] * 1.15)
    required_ram = int(profile.get("minimum_system_ram_bytes", 0))
    report = {
        **environment,
        "profile": profile_name,
        "model": profile["model"],
        "model_revision": profile["revision"],
        "estimated_download_bytes": profile["estimated_download_bytes"],
        "disk_free_bytes": disk.free,
        "required_disk_bytes": required_disk,
        "minimum_system_ram_bytes": required_ram,
        "checks": {
            "nvidia_l4": l4,
            "vllm_installed": vllm_installed,
            "vllm_checkpoint_support": vllm_checkpoint_support,
            "l4_kernel_capability": l4_kernel_capability,
            "disk_sufficient": disk.free >= required_disk,
            "ram_sufficient": environment["system_ram_bytes"] is not None
            and environment["system_ram_bytes"] >= required_ram,
        },
    }
    atomic_write_json(output, report)
    if strict and not all(report["checks"].values()):
        failures = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError("preflight failed before artifact download: " + ", ".join(failures))
    return report
