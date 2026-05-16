"""Class-aware primitive extraction.

The histogram-based extractors in ``cloud2bim/elements/`` look at *all*
points and have to guess which ones belong to walls vs furniture vs
floor. The ML pipeline gives us per-point class labels up front, so we
can run RANSAC / clustering only on points of the relevant class — much
less ambiguous.

This package is the v3 replacement for the histogram path. Each module
takes a SemanticLabels object plus the point cloud and returns the same
dataclasses (Slab, Wall, Opening) that the IFC builder already knows.

Pipeline mode dispatch lives in ``cloud2bim/pipeline.py`` and
``cloud2bim/stepwise.py`` — those decide whether to call into here or
fall back to the geometric extractors.
"""
from cloud2bim.extraction.slabs_ml import extract_slabs_ml
from cloud2bim.extraction.walls_ml import extract_walls_ml
from cloud2bim.extraction.openings_ml import extract_openings_ml

__all__ = [
    "extract_slabs_ml",
    "extract_walls_ml",
    "extract_openings_ml",
]
