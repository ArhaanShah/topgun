import sys

import pytest

from phase_a.preflight import run_preflight


@pytest.fixture
def base_environment():
    return {
        "python": sys.version,
        "platform": "test",
        "packages": {"vllm": "0.19.0", "torch": "2.0.0", "transformers": "4.30.0"},
        "system_ram_bytes": 40 * 1024**3,
        "gpu": [{"name": "NVIDIA L4", "memory_total_mib": 23000, "driver_version": "535", "compute_capability": "8.9"}],
        "cuda_runtime": "11.8",
        "torch_cuda_available": True,
    }


@pytest.fixture
def base_storage():
    return {
        "cache_exists": True,
        "runs_exists": True,
        "cache_writable": True,
        "runs_writable": True,
        "cache_inside_repository": False,
        "runs_inside_repository": False,
    }


def test_preflight_passes_with_sufficient_resources(tmp_path, base_environment, base_storage, monkeypatch):
    monkeypatch.setattr("phase_a.preflight.environment_report", lambda: base_environment)
    monkeypatch.setattr("phase_a.preflight.storage_report", lambda c, r: base_storage)
    monkeypatch.setattr(
        "phase_a.preflight._pinned_versions", lambda: {"vllm": "0.19.0", "torch": "2.0.0", "transformers": "4.30.0"}
    )

    class MockDiskUsage:
        free = 100 * 1024**3

    monkeypatch.setattr("shutil.disk_usage", lambda p: MockDiskUsage())

    monkeypatch.setattr("sys.version_info", (3, 11, 0))
    report = run_preflight("fp8_offload", tmp_path, tmp_path / "out.json", strict=False)
    assert all(report["checks"].values())


def test_preflight_fails_insufficient_disk(tmp_path, base_environment, base_storage, monkeypatch):
    monkeypatch.setattr("phase_a.preflight.environment_report", lambda: base_environment)
    monkeypatch.setattr("phase_a.preflight.storage_report", lambda c, r: base_storage)
    monkeypatch.setattr("phase_a.preflight._pinned_versions", lambda: {"vllm": "0.19.0"})

    class MockDiskUsage:
        free = 10 * 1024**3  # Only 10GB free

    monkeypatch.setattr("shutil.disk_usage", lambda p: MockDiskUsage())

    monkeypatch.setattr("sys.version_info", (3, 11, 0))
    report = run_preflight("fp8_offload", tmp_path, tmp_path / "out.json", strict=False)
    assert not report["checks"]["cache_disk_sufficient"]


def test_preflight_fails_insufficient_ram(tmp_path, base_environment, base_storage, monkeypatch):
    base_environment["system_ram_bytes"] = 16 * 1024**3  # 16GB
    monkeypatch.setattr("phase_a.preflight.environment_report", lambda: base_environment)
    monkeypatch.setattr("phase_a.preflight.storage_report", lambda c, r: base_storage)
    monkeypatch.setattr("phase_a.preflight._pinned_versions", lambda: {})

    class MockDiskUsage:
        free = 100 * 1024**3

    monkeypatch.setattr("shutil.disk_usage", lambda p: MockDiskUsage())

    monkeypatch.setattr("sys.version_info", (3, 11, 0))
    report = run_preflight("fp8_offload", tmp_path, tmp_path / "out.json", strict=False)
    assert not report["checks"]["ram_sufficient"]


def test_preflight_fails_wrong_gpu(tmp_path, base_environment, base_storage, monkeypatch):
    base_environment["gpu"] = [
        {"name": "NVIDIA T4", "memory_total_mib": 15000, "driver_version": "535", "compute_capability": "7.5"}
    ]
    monkeypatch.setattr("phase_a.preflight.environment_report", lambda: base_environment)
    monkeypatch.setattr("phase_a.preflight.storage_report", lambda c, r: base_storage)
    monkeypatch.setattr("phase_a.preflight._pinned_versions", lambda: {})

    class MockDiskUsage:
        free = 100 * 1024**3

    monkeypatch.setattr("shutil.disk_usage", lambda p: MockDiskUsage())

    monkeypatch.setattr("sys.version_info", (3, 11, 0))
    report = run_preflight("fp8_offload", tmp_path, tmp_path / "out.json", strict=False)
    assert not report["checks"]["exactly_one_nvidia_l4"]
    assert not report["checks"]["l4_vram_approximately_24gb"]
    assert not report["checks"]["l4_kernel_capability"]


def test_preflight_fails_missing_cuda(tmp_path, base_environment, base_storage, monkeypatch):
    base_environment["torch_cuda_available"] = False
    monkeypatch.setattr("phase_a.preflight.environment_report", lambda: base_environment)
    monkeypatch.setattr("phase_a.preflight.storage_report", lambda c, r: base_storage)
    monkeypatch.setattr("phase_a.preflight._pinned_versions", lambda: {})

    class MockDiskUsage:
        free = 100 * 1024**3

    monkeypatch.setattr("shutil.disk_usage", lambda p: MockDiskUsage())

    monkeypatch.setattr("sys.version_info", (3, 11, 0))
    report = run_preflight("fp8_offload", tmp_path, tmp_path / "out.json", strict=False)
    assert not report["checks"]["torch_cuda_available"]


def test_preflight_fails_path_not_writable(tmp_path, base_environment, base_storage, monkeypatch):
    base_storage["cache_writable"] = False
    monkeypatch.setattr("phase_a.preflight.environment_report", lambda: base_environment)
    monkeypatch.setattr("phase_a.preflight.storage_report", lambda c, r: base_storage)
    monkeypatch.setattr("phase_a.preflight._pinned_versions", lambda: {})

    class MockDiskUsage:
        free = 100 * 1024**3

    monkeypatch.setattr("shutil.disk_usage", lambda p: MockDiskUsage())

    monkeypatch.setattr("sys.version_info", (3, 11, 0))
    report = run_preflight("fp8_offload", tmp_path, tmp_path / "out.json", strict=False)
    assert not report["checks"]["cache_exists_and_writable"]
