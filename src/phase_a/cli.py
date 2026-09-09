"""Unified command line interface for the Phase A workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .environment import enable_offline_mode
from .run import default_cache_dir, default_runs_dir, initialize_manifest, resolve_run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A local reproduction pipeline")
    parser.add_argument(
        "--command",
        required=True,
        choices=(
            "preflight",
            "download",
            "verify-offline",
            "prepare",
            "select",
            "health-check",
            "smoke",
            "reproduce",
            "judge",
            "audit",
            "finalize",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", default="fp8_offload", choices=("fp8_offload", "awq_4bit"))
    parser.add_argument("--runs-dir", type=Path, default=default_runs_dir())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--run-id")
    parser.add_argument("--split", choices=("smoke", "reproduction"), default="smoke")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use deterministic CPU-only mock models")
    parser.add_argument("--online", action="store_true", help="Allow downloads during prepare (never inference)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "preflight":
        from .preflight import run_preflight

        report = run_preflight(args.profile, args.cache_dir, args.cache_dir / "preflight.json", strict=not args.dry_run)
        print(json.dumps(report, indent=2))
        return
    if args.command == "download":
        from scripts.download_artifacts import download

        print(json.dumps(download(args.profile, args.cache_dir, dry_run=args.dry_run), indent=2))
        return
    if args.command == "verify-offline":
        from scripts.verify_offline import verify

        print(json.dumps(verify(args.profile, args.cache_dir), indent=2))
        return
    create = args.command in {"prepare", "select"}
    run_dir = resolve_run_dir(args.runs_dir, args.run_id, create=create)
    if args.command in {"prepare", "select"}:
        from .prepare import build_mock_frames, prepare_run

        manifest_path = run_dir / "manifests" / "run_manifest.json"
        if not manifest_path.exists():
            initialize_manifest(
                run_dir, args.profile, mock=args.mock, allow_dirty=args.allow_dirty, offline=not args.online
            )
        frames = build_mock_frames() if args.mock else None
        prepare_run(run_dir, args.profile, args.cache_dir, mock_frames=frames, offline=not args.online)
        print(run_dir)
        return
    if args.command == "health-check":
        from .smoke_test import run_health_check

        if not args.mock:
            enable_offline_mode()
        print(
            json.dumps(
                run_health_check(
                    run_dir,
                    args.profile,
                    mock=args.mock,
                    dry_run=args.dry_run,
                    cache_dir=args.cache_dir,
                ),
                indent=2,
            )
        )
        return
    if args.command in {"smoke", "reproduce"}:
        from .generate import run_generation

        health_path = run_dir / "manifests" / "health_check.json"
        if not args.dry_run and (not health_path.exists() or json.loads(health_path.read_text())["status"] != "PASS"):
            raise RuntimeError("a passing health check is required before sampling")
        if not args.mock:
            enable_offline_mode()
        split = "smoke" if args.command == "smoke" else "reproduction"
        print(
            json.dumps(
                run_generation(
                    run_dir,
                    split,
                    profile_name=args.profile,
                    limit=args.limit,
                    dry_run=args.dry_run,
                    cache_dir=args.cache_dir,
                    resume=args.resume,
                    mock=args.mock,
                ),
                indent=2,
            )
        )
        return
    if args.command == "judge":
        from .judge import run_judge

        if not args.mock:
            enable_offline_mode()
        print(
            json.dumps(
                run_judge(
                    run_dir,
                    args.split,
                    limit=args.limit,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    mock=args.mock,
                    cache_dir=args.cache_dir,
                ),
                indent=2,
            )
        )
        return
    if args.command == "audit":
        from .audit import create_audit_package

        print(create_audit_package(run_dir))
        return
    if args.command == "finalize":
        from .finalize import finalize_run

        print(finalize_run(run_dir))


if __name__ == "__main__":
    main()
