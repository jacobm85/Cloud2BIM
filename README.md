# Cloud2BIM v2

Production Scan-to-BIM pipeline. Converts E57/LAS/LAZ/XYZ point clouds into
IFC 2x3 models with semantic ML pre-filtering, PCA-corrected wall detection,
sloped roof handling, and Revit-friendly output.

Fork of [VaclavNezerka/Cloud2BIM](https://github.com/VaclavNezerka/Cloud2BIM).
The v2 rewrite (this branch) replaces the original monolithic scripts with a
typed package, adds ML semantic segmentation, and addresses the geometric
edge-cases that triggered NaN crashes on real-world building scans.

---

## What's new in v2

| Capability | v1 | v2 |
|---|---|---|
| Input formats | E57 / LAS via converter step | E57 / LAS / LAZ / XYZ read directly |
| Coordinate handling | Fails silently on SWEREF | Auto-centers, preserves IFC site placement |
| Semantic segmentation | None | PointTransformer V3, RandLA-Net, or none |
| Wall detection | Trips on furniture | ML-filtered + RANSAC fallback |
| Roof detection | Horizontal slabs only | Sloped via RANSAC plane fitting |
| IFC output | Manual entity construction | ifcopenshell.api with stable GUIDs |
| Configuration | Untyped flat YAML | Pydantic-validated nested YAML |
| Error handling | Crashes whole job | Per-storey fault tolerance |

---

## Quick start (Docker)

```bash
mkdir -p /mnt/SSD250/scan2bim/{uploads,jobs,output_xyz,images,pointclouds}
docker compose up --build
```

Open **http://localhost:8001** and follow the 4-step wizard.

To enable ML semantic segmentation, mount a model weights directory and set
`segmentation.enabled: true` in the per-job config (or update the default in
`web/main.py`):

```yaml
# in docker-compose.yml volumes:
- /path/to/model/weights:/data/models:ro
```

Install ML dependencies inside the container:
```bash
docker compose exec cloud2bim pip install -r requirements-ml.txt
```

---

## CLI usage

```bash
python -m cloud2bim run config.yaml
python -m cloud2bim validate scan.e57         # sanity-check a point cloud
```

See `cloud2bim/config/defaults.yaml` for the full schema.

---

## Architecture

```
cloud2bim/
├── pipeline.py              orchestrator
├── config/
│   └── schema.py            pydantic Config (typed, validated)
├── io/
│   ├── readers.py           E57 / LAS / XYZ → numpy
│   └── coordinates.py       SWEREF-safe centering
├── segmentation/            ML semantic labelling
│   ├── base.py              Segmenter ABC + SemanticLabels
│   ├── ptv3.py              PointTransformer V3 (Pointcept)
│   ├── randla.py            RandLA-Net (Open3D-ML)
│   └── factory.py           backend dispatch w/ passthrough fallback
├── geometry/                pure geometric utilities
│   ├── lines.py             NaN-safe line intersection / distance
│   ├── pca.py               dominant-orientation detection
│   └── polygon.py           contour smoothing, polygon offsetting
├── elements/                building element extractors
│   ├── slabs.py             horizontal Z-histogram slab detection
│   ├── walls.py             2D-histogram + ML filter + NaN guards
│   ├── openings.py          windows / doors from wall cross-sections
│   └── roofs.py             RANSAC plane fitting for sloped roofs
├── ifc/
│   └── builder.py           IFC 2x3 with stable GUIDs (Revit-ready)
└── cli.py                   `python -m cloud2bim run|validate`

scripts/
└── validate.py              end-to-end smoke test with element counts

web/                         FastAPI + Three.js IFC viewer (unchanged)
```

---

## Plugging in custom-trained model weights

`SegmentationConfig` accepts a path to your own checkpoint:

```yaml
segmentation:
  backend: ptv3
  weights_path: /data/models/my-finetuned.pth
  wall_classes: [wall, partition_wall]      # match your label vocabulary
  clutter_classes: [furniture, equipment]
  door_classes: [door, sliding_door]
  window_classes: [window, skylight]
```

No code changes needed — the `Segmenter` interface absorbs label vocabulary
differences via config.

---

## Validation

Run the smoke test against any config:

```bash
python scripts/validate.py path/to/config.yaml
python scripts/validate.py path/to/config.yaml --reference reference.ifc
```

Reports timing per stage and element counts. With `--reference`, also diffs
element counts against a known-good IFC.

---

## License

Original research code by Václav Nežerka et al., CTU in Prague.
v2 production rewrite by Jacob.
