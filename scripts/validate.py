"""End-to-end pipeline validation.

Runs the full v2 pipeline against a test scan and reports:
    - Step timings
    - Element counts (slabs, walls, openings, roofs)
    - Output IFC file size
    - Optional comparison against a reference IFC (if provided)

Usage:
    python scripts/validate.py path/to/config.yaml
    python scripts/validate.py path/to/config.yaml --reference reference.ifc
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Cloud2BIM v2 pipeline validation")
    parser.add_argument("config", type=Path, help="Pipeline config YAML")
    parser.add_argument("--reference", type=Path, help="Optional reference IFC to compare against")
    args = parser.parse_args()

    # Ensure repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cloud2bim.config import load_config
    from cloud2bim.logging import configure, get_logger
    from cloud2bim.pipeline import run_pipeline

    configure(level="INFO")
    log = get_logger("validate")

    cfg = load_config(args.config)
    log.info("=" * 60)
    log.info("Cloud2BIM v2 — VALIDATION")
    log.info("Config: %s", args.config)
    log.info("Inputs: %s", [str(p) for p in cfg.io.input_files])
    log.info("=" * 60)

    t0 = time.time()
    rc = run_pipeline(cfg)
    elapsed = time.time() - t0

    log.info("=" * 60)
    log.info("RESULT")
    log.info("Total time: %.1f s", elapsed)
    log.info("Exit code: %d", rc)
    out = Path(cfg.io.output_ifc)
    if out.exists():
        size_mb = out.stat().st_size / 1_000_000
        log.info("IFC output: %s (%.2f MB)", out, size_mb)
        _report_ifc_contents(out, log)
    else:
        log.error("IFC NOT WRITTEN")

    if args.reference and args.reference.exists():
        log.info("─── Reference comparison ───")
        _compare_to_reference(out, args.reference, log)

    log.info("=" * 60)
    return rc


def _report_ifc_contents(path: Path, log) -> None:
    try:
        import ifcopenshell
    except ImportError:
        log.warning("ifcopenshell not available — skipping IFC inspection")
        return
    model = ifcopenshell.open(str(path))
    counts = {
        "Slabs": len(model.by_type("IfcSlab")),
        "Walls": len(model.by_type("IfcWallStandardCase")) + len(model.by_type("IfcWall")),
        "Doors": len(model.by_type("IfcDoor")),
        "Windows": len(model.by_type("IfcWindow")),
        "Openings": len(model.by_type("IfcOpeningElement")),
        "Roofs": len(model.by_type("IfcRoof")),
        "Storeys": len(model.by_type("IfcBuildingStorey")),
    }
    log.info("IFC element counts:")
    for k, v in counts.items():
        log.info("  %-10s %d", k + ":", v)


def _compare_to_reference(produced: Path, reference: Path, log) -> None:
    """Quick element-count comparison. Geometric Hausdorff is left as TODO."""
    try:
        import ifcopenshell
    except ImportError:
        return
    a = ifcopenshell.open(str(produced))
    b = ifcopenshell.open(str(reference))
    for cls in ("IfcSlab", "IfcWallStandardCase", "IfcDoor", "IfcWindow", "IfcRoof"):
        na, nb = len(a.by_type(cls)), len(b.by_type(cls))
        delta = na - nb
        log.info("  %-25s produced=%d reference=%d  delta=%+d", cls + ":", na, nb, delta)


if __name__ == "__main__":
    raise SystemExit(main())
