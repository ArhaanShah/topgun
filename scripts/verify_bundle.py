#!/usr/bin/env python3
"""Thin wrapper for :mod:`phase_a.bundle`."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_a.bundle import verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle), indent=2))


if __name__ == "__main__":
    main()
