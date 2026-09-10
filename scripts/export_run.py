#!/usr/bin/env python3
"""Thin wrapper for :mod:`phase_a.bundle`."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_a.bundle import export_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    bundle, checksum = export_run(args.run_dir, args.output_dir)
    print(bundle)
    print(checksum)


if __name__ == "__main__":
    main()
