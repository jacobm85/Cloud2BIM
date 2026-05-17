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
        description=(
            "Path to model weights. None = auto-download a pretrained "
            "S3DIS checkpoint for the chosen backend and cache it in "
            "%LOCALAPPDATA%/cloud2bim/models (Windows) or "
            "~/.cache/cloud2bim/models (Linux/Mac)."
        ),
    )
    ml_voxel_size: float = Field(
        default=0.05,
        gt=0,
        description=(
            "Voxel size for ML inference (m). Default 5 cm matches the "
            "S3DIS training resolution of the pretrained PTv3/RandLA "
            "weights — using a smaller voxel (e.g. 1 cm) makes inference "
            "much slower AND degrades accuracy because the model is "
            "out-of-distribution. Geometric precision in the final BIM "
            "is set by geometry_resolution, not this. Override only if "
            "your custom weights were trained at a different scale."
        ),
    )
    geometry_resolution: float = Field(
        default=0.01,
        gt=0,
        description=(
            "Resolution (m) for RANSAC plane fitting, slab boundary "
            "extraction and opening detection. This is what controls "
            "final BIM precision. Default 1 cm is appropriate for most "
            "high-density indoor scans."
        ),
    )
    has_rgb: Literal["auto", "true", "false"] = Field(
        default="auto",
        description=(
            "Whether to feed RGB to the model. 'auto' inspects the "
            "input file and uses RGB when available, falling back to "
            "height-above-floor as the single scalar feature. Override "
            "to 'false' if the cloud has RGB but the colour is noise "
            "(e.g. uniform grey from a structured-light scanner)."
        ),
    )
    device: Literal["cuda", "cpu", "auto"] = "auto"
    batch_size: int = Field(default=4, ge=1)
    max_voxels_per_batch: int = Field(
        default=60_000,
        ge=5_000,
        description=(
            "Chunk inference into batches of at most this many voxels. "
            "PTv3 with RPE builds large attention matrices; ~60k voxels "
            "fits comfortably on a 12 GB card and ~150k on a 24 GB card. "
            "Raise if you have headroom — chunking has a small accuracy "
            "cost from seam-voting between tiles."
        ),
    )
    cache_labels: bool = Field(
        default=True,
        description="Save labels.npy to work_dir so re-runs skip inference",
    )

    # Which S3DIS classes to treat as walls / clutter / openings etc.
    wall_classes: List[str] = Field(default=["wall"])
    floor_classes: List[str] = Field(default=["floor"])
    ceiling_classes: List[str] = Field(default=["ceiling"])
    column_classes: List[str] = Field(default=["column"])
    clutter_classes: List[str] = Field(default=["clutter", "chair", "table", "sofa", "bookcase"])
    door_classes: List[str] = Field(default=["door"])
    window_classes: List[str] = Field(default=["window"])

    # Backwards-compat: code still calling .voxel_size gets the ML voxel.
    @property
    def voxel_size(self) -> float:
        return self.ml_voxel_size


# ─── Slabs ────────────────────────────────────────────────────────────────────

class SlabConfig(BaseModel):
    enabled: bool = Field(default=True, description="Toggle slab detection off entirely")
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
    enabled: bool = Field(default=True, description="Toggle wall detection off entirely")
    min_length: float = Field(
        default=0.05,
        gt=0,
        description=(
            "Minimum wall length (m). Lowered from the old 0.10 default — "
            "real interior partitions often run shorter than 10 cm in "
            "scans (short stubs between door frames etc.). Raise if you "
            "see noise being promoted to walls."
        ),
    )
    min_thickness: float = Field(default=0.05, gt=0, description="m")
    max_thickness: float = Field(default=0.75, gt=0, description="m")
    exterior_thickness: float = Field(default=0.30, gt=0, description="m")
    collinear_merge_distance: float = Field(
        default=1.5,
        gt=0,
        description=(
            "m. Two parallel, near-collinear segments merge when their "
            "endpoints are within this distance. Decoupled from "
            "max_thickness so you can keep a tight thickness band while "
            "letting the contour-tracer's fragmented runs merge into one "
            "long wall. The merged result is re-queued so chains of "
            "fragments collapse iteratively."
        ),
    )
    pair_min_overlap: float = Field(
        default=0.20,
        gt=0,
        description=(
            "m. When pairing two parallel segments as the inner and "
            "outer face of one wall, they must overlap along the wall "
            "direction by at least this much. Decoupled from "
            "max_thickness so short walls (between door frames, kitchen "
            "islands) can still be detected as two-sided."
        ),
    )
    singleton_min_length: float = Field(
        default=0.30,
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
            "use the default 130-160 cm above-floor band. Wizard mode lets "
            "the user set these via the histogram slider."
        ),
    )
    cross_section_bands_lower: List[Optional[List[float]]] = Field(
        default_factory=list,
        description=(
            "Per-storey low-section Z-bands (default 30-35 cm above floor). "
            "Only used when require_lower_support is true: each detected "
            "wall must also have point support in this band, otherwise it "
            "is dropped — filters out window-as-wall confusions."
        ),
    )
    require_lower_support: bool = Field(
        default=False,
        description=(
            "If true, drop walls that don't also show up in the low "
            "cross-section band (cross_section_bands_lower). Real walls go "
            "from floor to ceiling, windows don't — so the low band sees "
            "walls but not windows, and walls without that support are "
            "likely misclassified windows or other tall narrow features."
        ),
    )
    lower_support_fraction: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of a detected wall's length that must overlap with "
            "low-section points for the wall to be kept. Tighter (0.5+) "
            "drops more walls; looser (0.1) keeps almost everything."
        ),
    )

    # ── Vertical-continuity algorithm parameters ────────────────────────
    #
    # Used only when algorithm="vertical". A wall is detected as an
    # XY-pixel column where points are present from floor to ceiling
    # without significant lateral drift between Z-slices.
    vertical_slice_thickness: float = Field(
        default=0.05,
        gt=0,
        description=(
            "m. Height of each Z-slice when scanning a pixel column "
            "from floor to ceiling. 5 cm matches typical scan density; "
            "raise to 10 cm for sparse clouds."
        ),
    )
    vertical_min_fill: float = Field(
        default=0.70,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of Z-slices a pixel must be filled in to count "
            "as a wall candidate. 0.7 = 70% of storey height. Lower "
            "the value to keep wall pixels near windows (where the "
            "column is empty over the glazing); raise it to drop "
            "furniture that only fills the lower half."
        ),
    )
    vertical_min_points_per_slice: int = Field(
        default=5,
        ge=1,
        description=(
            "Minimum points in a pixel × slice cell to count it as "
            "'filled'. Filters out noise — single stray points don't "
            "create a wall column."
        ),
    )
    vertical_sample_count: int = Field(
        default=5,
        ge=2,
        le=20,
        description=(
            "Number of sample heights between floor and ceiling for "
            "the sparse K-of-N variant. 5 covers 20/40/60/80/95% of "
            "the storey height — survives a single window band knocking "
            "out 1–2 samples. Raise for tall storeys, lower for low ones."
        ),
    )
    vertical_min_hits: int = Field(
        default=3,
        ge=1,
        description=(
            "Minimum number of sample heights at which a pixel must be "
            "filled to count as a wall. With sample_count=5, hits=3 means "
            'a wall must show up in at least 3 of the 5 height bands — '
            "tolerates fönsterband knocking out two contiguous bands."
        ),
    )
    vertical_pixel_size_cm: float = Field(
        default=5.0,
        gt=0,
        description=(
            "cm. XY pixel size for the v3 occupancy grid. Overrides the "
            "default pc_resolution × grid_coefficient (which gives 1 cm "
            "with typical settings — unnecessarily fine for a 10–30 cm "
            "thick wall and noisier than 5 cm). Raise for sparse clouds "
            "or to merge adjacent wall fragments; lower for very thin "
            "interior walls."
        ),
    )


# ─── Openings ─────────────────────────────────────────────────────────────────

class OpeningConfig(BaseModel):
    enabled: bool = Field(default=True, description="Toggle window/door detection off entirely")
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


# ─── Columns ──────────────────────────────────────────────────────────────────

class ColumnConfig(BaseModel):
    """Vertical free-standing column detection.

    Columns are small XY blobs that span (close to) the full storey height
    and aren't connected to any wall. Distinct from walls by their compact
    aspect ratio in both X and Y.
    """
    enabled: bool = Field(default=False)
    min_size: float = Field(default=0.15, gt=0, description="Minimum column edge (m)")
    max_size: float = Field(default=0.80, gt=0, description="Maximum column edge (m)")
    min_height_fraction: float = Field(
        default=0.7, gt=0, le=1.0,
        description="Fraction of storey height the column must span vertically",
    )
    min_points: int = Field(default=200, ge=10, description="Minimum points to consider a column")
    wall_clearance: float = Field(
        default=0.20, ge=0,
        description="Drop blobs within this distance of any detected wall axis",
    )


# ─── Stairs ───────────────────────────────────────────────────────────────────

class StairConfig(BaseModel):
    """Stair flight detection from a series of close-spaced horizontal peaks.

    Each step is a horizontal tread (~20-30 cm deep) and the runs sit
    between the main slab levels. A run is detected as 3+ peaks separated
    by `min_riser` .. `max_riser` along Z, with a compact common XY
    footprint (so we don't pick up e.g. balconies at different heights).
    """
    enabled: bool = Field(default=False)
    min_riser: float = Field(default=0.13, gt=0, description="Minimum step rise (m)")
    max_riser: float = Field(default=0.22, gt=0, description="Maximum step rise (m)")
    min_steps: int = Field(default=3, ge=2, description="Minimum steps in a flight")
    z_step: float = Field(
        default=0.03, gt=0,
        description="Z-histogram bin for step detection (much finer than slab z_step)",
    )
    peak_height_ratio: float = Field(
        default=0.05, gt=0, le=1.0,
        description="Peak threshold for step detection — lower than slabs since treads are sparser",
    )
    max_footprint: float = Field(
        default=4.0, gt=0,
        description="Maximum XY extent of a stair run (m). Filters out balconies.",
    )


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
    columns: ColumnConfig = Field(default_factory=ColumnConfig)
    stairs: StairConfig = Field(default_factory=StairConfig)
    roofs: RoofConfig = Field(default_factory=RoofConfig)
    ifc: IFCConfig = Field(default_factory=IFCConfig)

    exterior_scan: bool = False

    building_type: Literal["office", "industrial", "custom"] = Field(
        default="office",
        description=(
            "Controls the default cross-section Z-band used to detect "
            "walls when the user hasn't picked one manually. "
            "'office' = upstream's 85-120% of storey height (high band, "
            "clear of furniture; matches VaclavNezerka/Cloud2BIM). "
            "'industrial' = 25-35 cm above the floor (low band, clear "
            "of ceiling-mounted cable trays / ducts common along walls "
            "in industrial scans). "
            "'custom' = no preset; the wizard's per-storey band picker "
            "starts at a mid-wall fallback (130-160 cm absolute) and "
            "expects the user to set it explicitly."
        ),
    )

    algorithm: Literal["v1", "v2", "vertical"] = Field(
        default="v1",
        description=(
            "Wall + slab detection variant. 'v1' = original Cloud2BIM "
            "(VaclavNezerka/Cloud2BIM), kept as known-good baseline. "
            "'v2' = rewrite with geometric tweaks; use after verifying "
            "it beats v1 on your data. 'vertical' = vertical-continuity "
            "approach — a wall is any XY-pixel column that's filled "
            "from floor to ceiling. Robust against furniture and "
            "diagonal building orientations because each pixel is "
            "evaluated independently. Only consulted when pipeline_mode "
            "includes a geometric stage."
        ),
    )

    pipeline_mode: Literal["geometric", "hybrid", "ml"] = Field(
        default="hybrid",
        description=(
            "How extraction is run. 'geometric' = histogram-only (v1/v2 "
            "path, no ML required, fast but cluttered interiors degrade). "
            "'hybrid' (default) = ML-driven extraction with geometric "
            "fallback per storey when ML finds too few wall/floor points. "
            "'ml' = ML-only, no fallback — pick when the segmentation is "
            "known good and you want the cleanest separation from "
            "furniture. Segmentation is only loaded for hybrid/ml."
        ),
    )

    hybrid_min_class_points: int = Field(
        default=5_000,
        ge=100,
        description=(
            "Hybrid mode falls back to geometric extraction for any "
            "storey where the ML returns fewer than this many points "
            "for the relevant class (e.g. 'wall'). Lower = trust ML "
            "more aggressively; higher = trust geometric fallback more."
        ),
    )


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(**data)
