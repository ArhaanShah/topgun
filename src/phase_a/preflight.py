"""L4 resource and software preflight."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

from packaging.version import Version

from .config import REPO_ROOT, load_phase_config, load_profile, load_yaml
from .environment import SUPPORTED_PYTHON, environment_report, guard_local_inference, storage_report
from .storage import atomic_write_json

GIB = 1024**3


def _pinned_versions() -> dict[str, str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = data["project"]["optional-dependencies"]["gpu"] + data["project"]["dependencies"]
    wanted = {}
    for requirement in requirements:
        name = requirement.split("==", 1)[0].lower()
        if name in {"torch", "vllm", "transformers"} and "==" in requirement:
            wanted[name] = requirement.split("==", 1)[1].split(";", 1)[0].strip()
    return wanted


def run_preflight(
    profile_name: str,
    cache_dir: Path,
    output: Path,
    *,
    runs_dir: Path | None = None,
    strict: bool = True,
    allow_repo_storage: bool = False,
) -> dict[str, Any]:
    profile = load_profile(profile_name)
    guard_local_inference(profile)
    phase = load_phase_config()
    judge = load_yaml(REPO_ROOT / "configs" / "models" / "local_judge.yaml")
    runs_dir = runs_dir or cache_dir
    environment = environment_report()
    storage = storage_report(cache_dir, runs_dir)
    gpu_lines = environment["gpu"]
    exactly_one_l4 = len(gpu_lines) == 1 and "L4" in gpu_lines[0]["name"]
    l4_vram = exactly_one_l4 and 22000 <= gpu_lines[0]["memory_total_mib"] <= 25000
    pinned = _pinned_versions()
    versions_match = {
        name: environment["packages"].get(name, "not-installed") == version for name, version in pinned.items()
    }
    vllm_version = environment["packages"].get("vllm", "not-installed")
    vllm_checkpoint_support = vllm_version != "not-installed" and Version(vllm_version) >= Version("0.19.0")
    l4_kernel_capability = exactly_one_l4 and float(gpu_lines[0]["compute_capability"]) >= 8.9
    components = {
        "target_model": int(profile["estimated_download_bytes"]),
        "judge_model": int(judge.get("estimated_download_bytes", 6 * GIB)),
        "dataset": 2 * GIB,
        "python_gpu_environment": 12 * GIB,
        "download_overhead": 8 * GIB,
        "output_bundle": 4 * GIB,
    }
    calculated_disk = sum(components.values())
    required_disk = max(calculated_disk, 70 * GIB if profile_name == "fp8_offload" else calculated_disk)
    required_ram = int(profile.get("minimum_system_ram_bytes", 0))
    cache_free = None
    runs_free = None
    if storage["cache_exists"]:
        import shutil

        cache_free = shutil.disk_usage(cache_dir).free
    if storage["runs_exists"]:
        import shutil

        runs_free = shutil.disk_usage(runs_dir).free
    outside_repo = allow_repo_storage or not (storage["cache_inside_repository"] or storage["runs_inside_repository"])
    report = {
        **environment,
        "profile": profile_name,
        "model": profile["model"],
        "model_revision": profile["revision"],
        "estimated_download_bytes": profile["estimated_download_bytes"],
        "paths": storage,
        "disk_components_bytes": components,
        "cache_disk_free_bytes": cache_free,
        "runs_disk_free_bytes": runs_free,
        "required_disk_bytes": required_disk,
        "minimum_system_ram_bytes": required_ram,
        "recommended_system_ram_bytes": 40 * GIB if profile_name == "fp8_offload" else required_ram,
        "pinned_versions": pinned,
        "dataset_revision": phase["dataset"]["revision"],
        "judge_model": judge["model"],
        "judge_revision": judge["revision"],
        "checks": {
            "exactly_one_nvidia_l4": exactly_one_l4,
            "l4_vram_approximately_24gb": l4_vram,
            "torch_cuda_available": environment["torch_cuda_available"],
            "python_3_11": sys.version_info[:2] == SUPPORTED_PYTHON,
            **{f"{name}_version_pinned": passed for name, passed in versions_match.items()},
            "vllm_checkpoint_support": vllm_checkpoint_support,
            "l4_kernel_capability": l4_kernel_capability,
            "cache_exists_and_writable": storage["cache_exists"] and storage["cache_writable"],
            "runs_exists_and_writable": storage["runs_exists"] and storage["runs_writable"],
            "persistent_paths_outside_repo": outside_repo,
            "cache_disk_sufficient": cache_free is not None and cache_free >= required_disk,
            "runs_disk_sufficient": runs_free is not None and runs_free >= required_disk,
            "ram_sufficient": environment["system_ram_bytes"] is not None
            and environment["system_ram_bytes"] >= required_ram,
        },
    }
    atomic_write_json(output, report)
    if strict and not all(report["checks"].values()):
        failures = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError("preflight failed before artifact download: " + ", ".join(failures))
    return report
