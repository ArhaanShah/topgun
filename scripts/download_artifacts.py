#!/usr/bin/env python3
"""Thin wrapper for :mod:`phase_a.artifacts`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_a.artifacts import download
from phase_a.run import default_cache_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fp8_offload", "awq_4bit"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(args.profile, args.cache_dir, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
