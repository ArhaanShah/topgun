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
    raw_prompts = ["Name one primary color.", "Write one short sentence about rain."]
    if dry_run:
        return {"status": "DRY_RUN", "prompts": len(raw_prompts)}
    if mock:
        prompts = raw_prompts
        backend = backend or MockBackend()
    else:
        if cache_dir is None:
            raise ValueError("cache_dir is required for local production health checks")
        model_path = artifact_path(cache_dir, profile["model"], profile["revision"])
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        prompts = [render_user_prompt(tokenizer, text, profile["revision"]).rendered for text in raw_prompts]
        runtime_profile = dict(profile)
        runtime_profile["model"] = runtime_profile["tokenizer"] = str(model_path)
        backend = backend or make_backend(runtime_profile)
    # Two ordinary prompts, plus the first prompt again with a different seed.
    prompts.append(prompts[0])
    seeds = [derive_seed(run_dir.name, "health", i) for i in range(3)]
    started = time.monotonic()
    outputs = backend.generate(prompts, seeds, phase["generation"])
    nonempty = all(item.text.strip() for item in outputs)
    terminated = all(item.finish_reason in {"stop", "length"} for item in outputs)
    diverse = outputs[0].text != outputs[2].text
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
    peak_gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    report = {
        "status": "PASS" if all((nonempty, terminated, diverse, no_reasoning, interruption_resume)) else "FAIL",
        "nonempty": nonempty,
        "valid_termination": terminated,
        "different_seeds_differ": diverse,
        "reasoning_hidden": no_reasoning,
        "interruption_resume_no_duplicates": interruption_resume,
        "peak_gpu_memory_bytes": peak_gpu,
        "cpu_offload_gb": profile.get("cpu_offload_gb", 0),
        "responses_per_second": len(outputs) / max(time.monotonic() - started, 1e-9),
    }
    atomic_write_json(run_dir / "manifests" / "health_check.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("health check failed; no Phase A samples may be written")
    return report
