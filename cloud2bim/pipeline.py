"""Pipeline orchestrator.

Currently a stub; full implementation lands as each subsystem (segmentation,
slabs, walls, openings, roofs, ifc) gets its dedicated module.

Steps planned:
    1. Read point cloud
    2. Center coordinates (avoid SWEREF NaN)
    3. Optional: dilute
    4. Semantic segmentation → labels (cached per job)
    5. Slabs from horizontal histogram
    6. Walls from 2D histogram on wall-labelled subset
    7. Openings from wall cross-sections, validated against semantic labels
    8. Roofs from RANSAC plane fitting on ceiling labels
    9. IFC export
"""
from __future__ import annotations

from cloud2bim.config import Config
from cloud2bim.logging import get_logger

log = get_logger(__name__)


def run_pipeline(cfg: Config) -> int:
    """Run the full pipeline. Returns process exit code (0 = success)."""
    log.info("Pipeline scaffolding — implementation in progress")
    log.info(
        "Will process %d input file(s) → %s",
        len(cfg.io.input_files),
        cfg.io.output_ifc,
    )
    log.warning("run_pipeline() not yet implemented — see cloud2bim.pipeline")
    return 0
