"""Command-line entry point.

    python -m cloud2bim run config.yaml
    python -m cloud2bim validate scan.e57         (sanity-checks input)

The ``run`` subcommand orchestrates the full pipeline. The ``validate``
subcommand reads a point cloud and reports point count, bounds, and
estimated processing requirements without running the pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cloud2bim import __version__
from cloud2bim.config import load_config
from cloud2bim.logging import configure, get_logger


def cmd_run(args: argparse.Namespace) -> int:
    configure(level=args.log_level)
    log = get_logger("cloud2bim.cli")
    log.info("Cloud2BIM %s — pipeline run", __version__)

    cfg = load_config(args.config)
    log.info("Loaded config: %s", args.config)
    log.info("Inputs: %s", [str(p) for p in cfg.io.input_files])

    # Pipeline import is deferred to keep CLI startup fast and to avoid
    # heavy ML dependencies for ``validate`` runs.
    from cloud2bim.pipeline import run_pipeline

    return run_pipeline(cfg)


def cmd_validate(args: argparse.Namespace) -> int:
    configure(level=args.log_level)
    log = get_logger("cloud2bim.cli")

    from cloud2bim.io import read_pointcloud

    xyz, _ = read_pointcloud(args.input)
    log.info("Point count: %s", f"{len(xyz):,}")
    if len(xyz):
        mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
        log.info("Bounds X: %.2f .. %.2f", mins[0], maxs[0])
        log.info("Bounds Y: %.2f .. %.2f", mins[1], maxs[1])
        log.info("Bounds Z: %.2f .. %.2f", mins[2], maxs[2])
        if max(abs(mins[0]), abs(mins[1])) > 10_000:
            log.warning(
                "Large absolute coordinates detected — center_coordinates: true is required"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cloud2bim", description="Scan-to-BIM pipeline")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("config", type=Path, help="YAML config file")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="Sanity-check a point cloud")
    p_val.add_argument("input", type=Path, help="E57/LAS/LAZ/XYZ file")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
