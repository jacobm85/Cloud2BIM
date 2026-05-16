"""Abstract segmentation interface.

The pipeline only depends on this interface — concrete backends (PTv3,
RandLA-Net, custom-trained models) live behind it and can be swapped via
config without touching pipeline code.

To plug in your own trained weights:
    1. Set ``segmentation.backend`` to the matching backend
    2. Set ``segmentation.weights_path`` to your .pth/.ckpt
    3. Optionally adjust ``wall_classes`` etc. in config to match your
       label vocabulary
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# S3DIS label vocabulary used by default. Custom models can override these
# via SegmentationConfig.{wall,floor,...}_classes.
S3DIS_LABELS: tuple[str, ...] = (
    "ceiling", "floor", "wall", "beam", "column", "window",
    "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
)


@dataclass
class SemanticLabels:
    """Per-point semantic labels.

    ``label_ids`` is a numpy int array length N (one label per input point).
    ``label_names`` maps id → human-readable name (matches config classes).
    """
    label_ids: np.ndarray
    label_names: tuple[str, ...]

    def mask_for(self, classes: Sequence[str]) -> np.ndarray:
        """Boolean mask: True where the point belongs to any listed class."""
        wanted_ids = {i for i, name in enumerate(self.label_names) if name in classes}
        if not wanted_ids:
            return np.zeros(len(self.label_ids), dtype=bool)
        return np.isin(self.label_ids, list(wanted_ids))


class Segmenter(ABC):
    """Abstract per-point semantic segmenter."""

    @abstractmethod
    def segment(
        self, points: np.ndarray, rgb: np.ndarray | None = None
    ) -> SemanticLabels:
        """Return per-point labels.

        ``points`` is (N, 3) XYZ in world metres.
        ``rgb`` is (N, 3) colour in any range — implementations normalise.
        When ``rgb`` is None the implementation falls back to a synthetic
        feature (typically a constant grey) which works with S3DIS-pretrained
        weights at a small accuracy cost.
        """
        ...

    @property
    def label_vocabulary(self) -> tuple[str, ...]:
        """Override if your model uses a different label set."""
        return S3DIS_LABELS


class PassthroughSegmenter(Segmenter):
    """No segmentation — every point gets the same generic label.

    Used when ``segmentation.enabled = false`` so the rest of the pipeline
    can run without ML dependencies installed. Walls module falls back to
    using all points (legacy v1 behaviour).
    """

    LABEL_NAME = "unknown"

    def segment(
        self, points: np.ndarray, rgb: np.ndarray | None = None
    ) -> SemanticLabels:
        return SemanticLabels(
            label_ids=np.zeros(len(points), dtype=np.int32),
            label_names=(self.LABEL_NAME,),
        )

    @property
    def label_vocabulary(self) -> tuple[str, ...]:
        return (self.LABEL_NAME,)


def load_cached_labels(path: Path) -> SemanticLabels | None:
    """Load labels.npy from disk if it exists."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True).item()
    return SemanticLabels(label_ids=data["ids"], label_names=tuple(data["names"]))


def save_cached_labels(labels: SemanticLabels, path: Path) -> None:
    """Persist labels for re-runs (segmentation is the slowest step)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, {"ids": labels.label_ids, "names": list(labels.label_names)}, allow_pickle=True)
