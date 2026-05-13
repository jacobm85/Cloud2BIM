# Cloud2BIM

Automated Scan-to-BIM pipeline that converts 3D point clouds into IFC models.
Supports indoor building scans in E57, LAS/LAZ, and XYZ format.

Fork of [VaclavNezerka/Cloud2BIM](https://github.com/VaclavNezerka/Cloud2BIM) —
extended with a Docker-based web interface, additional input format support,
and various pipeline improvements.

---

## What it does

1. **Slab detection** — scans the Z-axis histogram to find floor and ceiling slabs
2. **Wall segmentation** — builds a 2D occupancy map, extracts wall contours and groups parallel segments; PCA rotation handles non-axis-aligned buildings
3. **Opening detection** — classifies windows and doors from wall cross-sections
4. **IFC export** — writes a standards-compliant IFC 2x3 file with slabs, walls, openings and spaces
5. **3D viewer** — interactive BIM viewer in the browser (Three.js)

---

## Quick start — Docker (recommended)

### 1. Create data directories

```bash
mkdir -p /mnt/ssd250/scan2bim/{uploads,jobs,output_xyz,images,pointclouds}
```

### 2. Edit `docker-compose.yml`

Change the host paths on the left side of `:` to match your setup:

```yaml
volumes:
  - /mnt/ssd250/scan2bim/uploads:/app/web/uploads
  - /mnt/ssd250/scan2bim/jobs:/app/web/jobs
  - /mnt/ssd250/scan2bim/output_xyz:/app/output_xyz
  - /mnt/ssd250/scan2bim/images:/app/images
  - /mnt/ssd250/scan2bim/pointclouds:/drives/Punktmoln
```

Add more `/drives/<Name>` lines for additional point cloud locations.

### 3. Build and run

```bash
docker compose up --build
```

Open **http://localhost:8001** in a browser.

---

## Web interface

The 4-step wizard guides you through:

1. **Select file** — upload a point cloud (E57, LAS, LAZ, XYZ) or browse a mounted network drive
2. **Parameters** — slab thickness, wall dimensions, dilution factor, IFC metadata
3. **Processing** — live log stream; jobs run in the background
4. **Result** — interactive 3D IFC viewer, download IFC file

### Network drives

Mount any host directory under `/drives/<Name>` in `docker-compose.yml`:

```yaml
- /mnt/nas/projects:/drives/NAS_Projekt:ro
- /mnt/nas/scans:/drives/Skanningar:ro
```

The name after `/drives/` appears automatically in the file browser — no other configuration needed.

---

## Command-line usage

```bash
pip install -r requirements.txt
python cloud2entities.py config.yaml
```

Edit `config.yaml` to point to your input files and set processing parameters.

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `dilute` | `true` | Enable point cloud downsampling |
| `dilution_factor` | `10` | Keep every Nth point (10 = 90% reduction) |
| `pc_resolution` | `0.002` | Expected point spacing (m) |
| `bfs_thickness` | `0.3` | Bottom floor slab thickness (m) |
| `tfs_thickness` | `0.4` | Top floor slab thickness (m) |
| `min_wall_length` | `0.10` | Minimum wall length to keep (m) |
| `max_wall_thickness` | `0.75` | Maximum wall thickness (m) |

---

## Supported formats

| Format | Notes |
|---|---|
| `.xyz` | Tab-separated ASCII, header line `//X\tY\tZ` |
| `.e57` | Native E57 — converted to XYZ automatically |
| `.las` / `.laz` | LiDAR formats — converted via laspy |

---

## Requirements

- Python 3.11+
- See `requirements.txt` (CPU-only) or `requirements-docker.txt` (Docker image)
- No GPU required

---

## Project structure

```
cloud2entities.py     Main pipeline script
aux_functions.py      Slab/wall/opening detection algorithms
space_generator.py    Room/zone extraction (Shapely)
generate_ifc.py       IFC model builder (ifcopenshell)
plotting_functions.py Debug visualisations
config.yaml           Default CLI configuration
web/
  main.py             FastAPI backend
  job_manager.py      Background job runner
  static/             Frontend (vanilla JS, Three.js viewer)
```

---

## License

Original research code by Václav Nežerka et al., CTU in Prague.
Web interface and pipeline improvements by Jacob.
