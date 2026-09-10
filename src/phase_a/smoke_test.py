"""Target backend health checks run before scientific samples."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from .config import artifact_path, load_phase_config, load_profile
from .inference import Backend, MockBackend, derive_seed, make_backend, render_user_prompt
from .storage import JSONLStorage, atomic_write_json


class HealthProbe(BaseModel):
    run_id: str
    split: str
    pattern_id: str
    sample_index: int
    value: str


def run_health_check(
    run_dir: Path,
    profile_name: str,
    *,
    backend: Backend | None = None,
    mock: bool = False,
    dry_run: bool = False,
    cache_dir: Path | None = None,
) -> dict:
    profile, phase = load_profile(profile_name), load_phase_config()
    from .run import load_manifest

    manifest = load_manifest(run_dir)
    if manifest.model_profile != profile_name:
        raise RuntimeError("profile cannot change inside an existing run; create a new run ID")
    raw_prompt = "Invent a vivid six-word sentence containing a color, an animal, and a number."
    if dry_run:
        return {"status": "DRY_RUN", "prompts": 1}
    load_started = time.monotonic()
    if mock:
        prompts = [raw_prompt]
        backend = backend or MockBackend()
    else:
        if cache_dir is None:
            raise ValueError("cache_dir is required for local production health checks")
        model_path = artifact_path(cache_dir, profile["model"], profile["revision"])
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        prompts = [render_user_prompt(tokenizer, raw_prompt, profile["revision"]).rendered]
        runtime_profile = dict(profile)
        runtime_profile["model"] = runtime_profile["tokenizer"] = str(model_path)
        backend = backend or make_backend(runtime_profile)
    load_time = time.monotonic() - load_started
    seeds = [derive_seed(run_dir.name, "health", 0)]
    peak_gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    started = time.monotonic()
    try:
        outputs = backend.generate(prompts, seeds, phase["generation"])
        failure = None
    except Exception as exc:
        outputs = []
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    generation_time = time.monotonic() - started
    nonempty = all(item.text.strip() for item in outputs)
    terminated = len(outputs) == 1 and all(item.finish_reason in {"stop", "length"} for item in outputs)
    decodable = len(outputs) == 1 and isinstance(outputs[0].text, str)
    no_reasoning = all("<think>" not in item.text.lower() and "</think>" not in item.text.lower() for item in outputs)
    probe_path = run_dir / "health_resume_probe.jsonl"
    probe = JSONLStorage(probe_path, HealthProbe)
    first = HealthProbe(run_id=run_dir.name, split="health", pattern_id="probe", sample_index=0, value="first")
    second = HealthProbe(run_id=run_dir.name, split="health", pattern_id="probe", sample_index=1, value="second")
    probe.append_record(first)
    # Reopening simulates termination after the first durable completion.
    resumed = JSONLStorage(probe_path, HealthProbe)
    resumed.append_record(first)
    resumed.append_record(second)
    interruption_resume = len(resumed.load_valid_records()) == 2
    try:
        import torch

        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    cpu_ram = None
    try:
        import psutil

        cpu_ram = psutil.Process().memory_info().rss
    except ImportError:
        pass
    passed = all((failure is None, nonempty, terminated, decodable, no_reasoning, interruption_resume))
    report = {
        "status": "PASS" if passed else "FAIL",
        "nonempty": nonempty,
        "decodable": decodable,
        "valid_termination": terminated,
        "reasoning_hidden": no_reasoning,
        "interruption_resume_no_duplicates": interruption_resume,
        "load_time_seconds": load_time,
        "generation_time_seconds": generation_time,
        "peak_gpu_memory_bytes": peak_gpu,
        "cpu_ram_bytes": cpu_ram,
        "cpu_offload_gb": profile.get("cpu_offload_gb", 0),
        "prompt_tokens": outputs[0].prompt_tokens if outputs else None,
        "completion_tokens": outputs[0].completion_tokens if outputs else None,
        "produced_text": outputs[0].text if outputs else "",
        "sampling_parameters": phase["generation"],
        "error": failure,
    }
    atomic_write_json(run_dir / "manifests" / "health_check.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(
            "health check failed; stop this run, create a new run ID, and try the pinned awq_4bit profile; never mix profiles"
        )
    return report
