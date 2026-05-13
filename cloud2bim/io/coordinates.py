"""Coordinate transformations.

Real-world coordinate systems (SWEREF 99, RT90, UTM) place buildings at
absolute coordinates with magnitudes around 1e6. The cross-product math
in line intersection (a*d - b*c with a,b,c,d ~ 1e6) loses precision and
produces NaN. We translate to a local origin before processing and
remember the offset so IFC output can place the building correctly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoordinateOffset:
    """XYZ translation applied to the point cloud."""

    x: float
    y: float
    z: float

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def apply_inverse(self, points: np.ndarray) -> np.ndarray:
        """Translate local coordinates back to the original frame."""
        return points + self.vector


def center_xy(points: np.ndarray) -> tuple[np.ndarray, CoordinateOffset]:
    """Subtract XY minimum so the cloud starts near the origin in X and Y.

    Z is preserved relative (offset.z = 0) so floor heights stay meaningful
    and the IFC site elevation remains valid.

    Returns the translated points and the offset that was applied.
    """
    if points.size == 0:
        return points, CoordinateOffset(0.0, 0.0, 0.0)
    offset = CoordinateOffset(
        x=float(points[:, 0].min()),
        y=float(points[:, 1].min()),
        z=0.0,
    )
    return points - offset.vector, offset
