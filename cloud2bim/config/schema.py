"""Strict typed configuration.

The pipeline never touches raw dicts — everything goes through Config which
fails loudly on missing/wrong keys at load time, not deep inside a 7-minute
slab segmentation run.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ─── I/O ──────────────────────────────────────────────────────────────────────

class IOConfig(BaseModel):
    """Where the point cloud comes from and where outputs go."""

    input_files: List[Path] = Field(..., description="One or more E57/LAS/LAZ/XYZ paths")
    output_ifc: Path = Field(..., description="Target IFC path")
    work_dir: Path = Field(default=Path("."), description="Intermediate file location")
    dilute: bool = True
    dilution_factor: int = Field(default=10, ge=1, description="Keep every Nth point")
    center_coordinates: bool = Field(
        default=True,
        description="Subtract XY minimum before processing (required for SWEREF etc.)",
    )

    @field_validator("input_files", mode="before")
    @classmethod
    def _wrap_single(cls, v):
        if isinstance(v, (str, Path)):
            return [v]
        return v


# ─── Semantic segmentation (ML) ──────────────────────────────────────────────

class SegmentationConfig(BaseModel):
    """ML semantic segmentation settings.

    Backend can be swapped without changing the rest of the pipeline.
    Custom-trained weights are loaded via ``weights_path``.
    """

    enabled: bool = True
    backend: Literal["ptv3", "randla", "none"] = "ptv3"
    weights_path: Optional[Path] = Field(
        default=None,
        description="Path to model weights. None = use bundled default for backend.",
    )
    voxel_size: float = Field(default=0.05, gt=0, description="Voxel size for ML input (m)")
    device: Literal["cuda", "cpu", "auto"] = "auto"
    batch_size: int = Field(default=4, ge=1)
    cache_labels: bool = Field(
        default=True,
        description="Save labels.npy to work_dir so re-runs skip inference",
    )

    # Which S3DIS classes to treat as walls / clutter / openings etc.
    wall_classes: List[str] = Field(default=["wall"])
    floor_classes: List[str] = Field(default=["floor"])
    ceiling_classes: List[str] = Field(default=["ceiling"])
    clutter_classes: List[str] = Field(default=["clutter", "chair", "table", "sofa", "bookcase"])
    door_classes: List[str] = Field(default=["door"])
    window_classes: List[str] = Field(default=["window"])


# ─── Slabs ────────────────────────────────────────────────────────────────────

class SlabConfig(BaseModel):
    bottom_floor_thickness: float = Field(default=0.3, gt=0, description="m")
    top_floor_thickness: float = Field(default=0.4, gt=0, description="m")
    pc_resolution: float = Field(default=0.002, gt=0, description="Expected point spacing (m)")
    grid_coefficient: int = Field(default=5, ge=1, description="Pixel = pc_resolution × this")
    z_step: float = Field(default=0.15, gt=0, description="Histogram bin width along Z (m)")
    max_slab_thickness: float = Field(
        default=0.5,
        gt=0,
        description=(
            "m. Adjacent Z-peaks closer than this are paired as bottom+top of one "
            "slab; peaks further apart become separate slabs (each treated as a "
            "floor surface). 50 cm covers typical RC slabs + finishes."
        ),
    )
    peak_height_ratio: float = Field(
        default=0.25,
        gt=0,
        le=1.0,
        description=(
            "Z-histogram peaks below this fraction of the max bin are ignored. "
            "Lower = pick up sparser ceilings; higher = avoid spurious peaks."
        ),
    )


# ─── Walls ────────────────────────────────────────────────────────────────────

class WallConfig(BaseModel):
    min_length: float = Field(default=0.10, gt=0, description="m")
    min_thickness: float = Field(default=0.05, gt=0, description="m")
    max_thickness: float = Field(default=0.75, gt=0, description="m")
    exterior_thickness: float = Field(default=0.30, gt=0, description="m")
    singleton_min_length: float = Field(
        default=0.50,
        gt=0,
        description=(
            "m. Minimum length for a one-faced segment (no parallel partner) "
            "to still count as a wall. Real exterior walls show up as long "
            "singletons because the scan only sees the inside face — keep this "
            "low enough to retain them but high enough to drop fragments from "
            "furniture or partial scans."
        ),
    )
    singleton_thickness: float = Field(
        default=0.30,
        gt=0,
        description="m. Default thickness assigned to one-faced (singleton) walls.",
    )
    use_ml_filter: bool = Field(
        default=True,
        description="Pre-filter pointcloud to wall_classes before histogram",
    )
    enable_ransac_fallback: bool = Field(
        default=True,
        description="Try 3D plane RANSAC for walls the 2D histogram misses (curved walls)",
    )
    max_walls_per_storey: int = Field(
        default=300,
        ge=1,
        description="Safety cap. Real scans on a single floor easily hit 100+ walls; the old 50 cap was throttling real walls.",
    )
    placeholder_height: float = Field(
        default=0.10,
        gt=0,
        description=(
            "Fallback wall height (m) when floor/ceiling can't be paired. "
            "Visible-but-short stubs so the modeller sees something is there."
        ),
    )
    cross_section_bands: List[Optional[List[float]]] = Field(
        default_factory=list,
        description=(
            "Per-storey absolute Z-band overrides for the wall cross-section. "
            "Each entry is [z_min, z_max] in metres (world Z); empty/null = "
            "use the default 30-130 cm above-floor band. Wizard mode lets "
            "the user set these via the histogram slider."
        ),
    )


# ─── Openings ─────────────────────────────────────────────────────────────────

class OpeningConfig(BaseModel):
    min_window_width: float = Field(default=0.40, gt=0)
    min_window_height: float = Field(default=0.60, gt=0)
    door_min_height: float = Field(default=1.60, gt=0)
    door_max_z: float = Field(default=0.10, ge=0, description="Threshold height (m)")
    max_aspect_ratio: float = Field(default=4.0, gt=0)
    validate_with_semantics: bool = Field(
        default=True,
        description="Reject candidate openings where points behind are furniture",
    )


# ─── Roofs ────────────────────────────────────────────────────────────────────

class RoofConfig(BaseModel):
    enabled: bool = False
    min_slope_deg: float = Field(default=10.0, ge=0, description="Below this = flat slab")
    ransac_distance: float = Field(default=0.05, gt=0, description="RANSAC inlier threshold (m)")
    min_inliers: int = Field(default=1000, ge=1)


# ─── IFC ──────────────────────────────────────────────────────────────────────

class IFCAuthor(BaseModel):
    given_name: str = ""
    family_name: str = ""
    organization: str = ""


class IFCProject(BaseModel):
    name: str = "Cloud2BIM Project"
    long_name: str = "Scan to BIM"
    version: str = "1.0"


class IFCBuilding(BaseModel):
    name: str = ""
    type: str = ""
    phase: str = ""


class IFCSite(BaseModel):
    latitude: List[int] = Field(default_factory=lambda: [0, 0, 0])
    longitude: List[int] = Field(default_factory=lambda: [0, 0, 0])
    elevation: float = 0.0


class IFCConfig(BaseModel):
    project: IFCProject = Field(default_factory=IFCProject)
    author: IFCAuthor = Field(default_factory=IFCAuthor)
    building: IFCBuilding = Field(default_factory=IFCBuilding)
    site: IFCSite = Field(default_factory=IFCSite)
    default_material: str = "Concrete"
    revit_compatible: bool = Field(
        default=True,
        description="Apply GUID stability and Revit-import hints",
    )


# ─── Root ─────────────────────────────────────────────────────────────────────

class Config(BaseModel):
    """Top-level pipeline configuration."""

    io: IOConfig
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    slabs: SlabConfig = Field(default_factory=SlabConfig)
    walls: WallConfig = Field(default_factory=WallConfig)
    openings: OpeningConfig = Field(default_factory=OpeningConfig)
    roofs: RoofConfig = Field(default_factory=RoofConfig)
    ifc: IFCConfig = Field(default_factory=IFCConfig)

    exterior_scan: bool = False


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(**data)
