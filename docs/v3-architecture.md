# Cloud2BIM v3 — ML-first Scan-to-BIM Architecture

Status: in-progress on branch `v3-ml` (branched from `v2-rewrite`).

## Why v3

v1/v2 detect walls and slabs from 2D/Z histograms. That works on a clean
empty building, but real interior scans are full of furniture, partitions
and clutter that show up in the histograms as wall-like density —
producing spurious walls and missed real ones.

v3 puts a semantic segmentation model *in front* of the geometric
extraction. Every point first gets a class label (`wall`, `floor`,
`ceiling`, `door`, `window`, `chair`, `table`, ...). Geometric
primitives are then fitted only to points of the relevant class.

## Pipeline shape

```
[1] Read points (E57/LAS/XYZ)  →  xyz (+ rgb if available)
[2] Dilution + coordinate centering
[3] RGB auto-detection
[4] Semantic segmentation         ← ML, replaces "every point is wall"
       voxelise → infer → NN-transfer labels back
[5] Class-aware primitive fitting ← replaces histogram extraction
       floor/ceiling points → RANSAC horizontal plane → Slab
       wall points          → DBSCAN clusters → vertical plane RANSAC → Wall
       door/window points    → AABB inside host wall → Opening
[6] Geometric refinement (snap intersections, orthogonality)
[7] IFC export                    ← reused from v2
[8] Floor-plan preview            ← reused from v2
```

## Pipeline modes

User chooses in the wizard:

- **`geometric`** — current v1/v2 histogram path. No ML required. Fast,
  but degrades on cluttered interiors.
- **`hybrid`** *(default)* — try ML extraction. For storeys where ML
  returns too few wall/floor points (configurable threshold), fall back
  to geometric. Best resilience.
- **`ml`** — ML-only, no histogram fallback. Use when you trust the
  segmentation and want the cleanest possible separation from furniture.

## Voxel size vs geometry resolution

Two independent settings, often confused:

| Setting | Default | What it controls |
| --- | --- | --- |
| `ml_voxel_size` | **0.05 m** | Voxel size for ML inference. PTv3/RandLA S3DIS weights are trained at 5 cm — using 1 cm gives slower inference *and worse* accuracy (out-of-distribution). |
| `geometry_resolution` | **0.01 m** | Resolution used for RANSAC plane fitting, slab boundary extraction, opening detection. This is where final BIM precision is set. |

You do **not** lose 1 cm precision in the BIM by running ML at 5 cm —
labels are NN-transferred back to the original full-resolution points,
and geometric primitives are fitted at `geometry_resolution`.

## RGB awareness

Pretrained S3DIS weights were trained on RGB-coloured point clouds. We:

1. Auto-detect RGB presence in the input (`xyz, rgb = read_pointcloud(...)`)
2. If RGB is present → feed `[r, g, b]` as point features (best accuracy)
3. If RGB is absent → feed `[height_above_floor]` as a single feature
   (compromise; accuracy drops by ~5–10 % vs RGB but still much better
   than nothing)
4. Expose `cfg.segmentation.has_rgb` as `auto | true | false` so the
   user can override the autodetection from the wizard.

## Hardware

Default `device = "auto"` resolves to CUDA when available, CPU otherwise.
PTv3 inference on a 50M-point scan takes ~30 s on a modern GPU vs
~10 min on CPU — encourage GPU.

## Class vocabulary (S3DIS)

```
ceiling | floor | wall | beam | column | window | door |
table | chair | sofa | bookcase | board | clutter
```

Mapping from class → element type lives in `SegmentationConfig`:

- `wall_classes = ["wall"]`
- `floor_classes = ["floor"]`
- `ceiling_classes = ["ceiling"]`
- `door_classes = ["door"]`
- `window_classes = ["window"]`
- `column_classes = ["column"]`
- `clutter_classes = ["clutter", "chair", "table", "sofa", "bookcase"]`

User can edit these from the wizard if a custom-trained model uses a
different vocabulary.

## Backends

| Backend | Status | Dependencies | Notes |
| --- | --- | --- | --- |
| `ptv3` | primary | torch + spconv + Pointcept | Best accuracy; needs CUDA |
| `randla` | fallback | torch + open3d-ml | Lighter; runs on CPU |
| `none`  | passthrough | none | All points → "unknown"; pipeline falls back to geometric |

Weights auto-download from public release URLs on first use, cached in
`%LOCALAPPDATA%\cloud2bim\models` (Windows) or `~/.cache/cloud2bim/models`
(Linux/Mac).

## What's reused from v2

- `cloud2bim/io/` — readers, coordinate centering
- `cloud2bim/ifc/` — IFC builder
- `cloud2bim/elements/` — Slab/Wall/Opening/Column/Stair dataclasses
- `cloud2bim/legacy/` — v1 implementations as fallback
- `cloud2bim/preview.py` — floor-plan PNG
- `web/` — wizard, viewer, job manager (extended with segmentation step)

## What's new

- `cloud2bim/extraction/` — class-aware primitive fitting
- `cloud2bim/segmentation/` — fix the existing ML scaffold so it
  actually runs end-to-end (Windows paths, RGB features, correct
  batching, weights download)
- Wizard step: model picker, voxel sliders, segmentation preview
