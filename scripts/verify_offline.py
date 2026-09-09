#!/usr/bin/env python3
"""Prove pinned tokenizer/config loading succeeds with sockets disabled."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_a.config import REPO_ROOT, load_phase_config, load_profile, load_yaml
from phase_a.environment import enable_offline_mode
from phase_a.run import default_cache_dir


def verify(profile_name: str, cache_dir: Path) -> dict:
    profile = load_profile(profile_name)
    judge = load_yaml(REPO_ROOT / "configs" / "models" / "local_judge.yaml")
    phase = load_phase_config()
    enable_offline_mode()
    original = socket.socket.connect

    def blocked(*args, **kwargs):
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
        dataset_root = cache_dir / "datasets" / "Transluce--WeirdChat" / phase["dataset"]["revision"]
        from datasets import load_from_disk

        for subset in ("patterns", "prompts", "rubrics"):
            load_from_disk(dataset_root / subset)
            checked.append({"role": f"dataset:{subset}", "path": str(dataset_root / subset)})
        from phase_a.inference import MockBackend

        probe = MockBackend().generate(["offline probe"], [1], {"max_tokens": 1})
        if not probe or not probe[0].text:
            raise RuntimeError("offline mock initialization probe failed")
        checked.append({"role": "one-token-mock-probe", "path": "in-process"})
    finally:
        socket.socket.connect = original
    return {"offline": True, "network_calls": 0, "checked": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fp8_offload", "awq_4bit"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    args = parser.parse_args()
    print(json.dumps(verify(args.profile, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()
