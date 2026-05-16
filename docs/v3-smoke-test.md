# v3 ML Pipeline — Smoke Test Guide

How to verify the v3-ml branch actually runs end-to-end on a real scan,
with the ML stack producing labels and class-aware extractors turning
them into IFC geometry.

Target scan for this team: `\\192.168.2.79\ssd250\scan2bim\pointclouds\vasakronan temp.e57`

> **Recommended path** — use the Docker images, not local pip.
> See "Docker workflow (recommended)" below before the pip section.

## Docker workflow (recommended)

Two images / two compose files:

| File | Purpose | Hardware |
| --- | --- | --- |
| `Dockerfile` + `docker-compose.yml` | Existing CPU-only image | Any host, no ML |
| `Dockerfile.ml` + `docker-compose.ml.yml` | **v3 ML image** | CUDA 12.x host with `nvidia-container-toolkit` |

### Build

The CUDA 12.4 base image works for any CUDA 12.x driver (forward-binary
compatible). Building doesn't need a GPU — only running does.

```bash
cd /opt/Cloud2BIM
git fetch origin && git checkout v3-ml && git pull
docker compose -f docker-compose.ml.yml build
```

Expect ~10 minutes the first time: pulls the 4 GB pytorch base image,
installs spconv-cu120 and Pointcept from source. Subsequent rebuilds
hit the layer cache.

### Point the container at your scans

`docker-compose.ml.yml` uses the same volume layout as the v2
`docker-compose.yml`:

- `/app/web/{uploads,jobs}`, `/app/output_xyz`, `/app/images` — local
  data dirs, edit the left side of `:` to your host paths.
- `/drives/<Namn>` — anything mounted here is auto-discovered by the
  wizard's file browser and shown with `<Namn>` as the display label
  (the scan logic lives in `web/main.py` at `_DRIVES_ROOT`).

Default mounts in the file:

```yaml
- /mnt/SSD250/scan2bim/pointclouds:/drives/Punktmoln
# - /mnt/annanNas/projekt:/drives/NAS_Projekt:ro
# - /mnt/annanNas/skanningar:/drives/Skanningar:ro
```

If the share isn't mounted on the host yet, the standard CIFS mount
works:

```bash
sudo mkdir -p /mnt/SSD250/scan2bim/pointclouds
sudo mount -t cifs //192.168.2.79/ssd250/scan2bim/pointclouds \
    /mnt/SSD250/scan2bim/pointclouds \
    -o ro,vers=3.0,credentials=/etc/cifs-credentials,uid=$(id -u),gid=$(id -g)
# Add to /etc/fstab for boot-time mounting.
```

The container reads from the mount directly — no file copying.

### (Optional) Pre-download the PTv3 weights

Auto-download happens on first inference. If your network can't reach
Hugging Face from the container, or you want to avoid the 555 MB
download interrupting the first run:

```bash
# Same host path as the bind-mount in docker-compose.ml.yml
mkdir -p /mnt/SSD250/scan2bim/models
cd /mnt/SSD250/scan2bim/models

# PointTransformer V3 — S3DIS Area 5, v3m1-0-rpe variant (555 MB)
curl -L -o ptv3-s3dis-area5.pth \
  "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth"

# Verify size
ls -lh ptv3-s3dis-area5.pth
# -rw-r--r-- 1 user user 555M ... ptv3-s3dis-area5.pth
```

The filename must be exactly `ptv3-s3dis-area5.pth` — that's what
`cloud2bim/segmentation/weights.py` REGISTRY looks for. If you grab
a different variant, rename it to that.

#### A note on the "Unsafe" pickle warning on Hugging Face

PyTorch `.pth` files are pickle-based, and Hugging Face flags every
pickle file as "Unsafe" because pickle can execute arbitrary Python at
load time. That warning is about the *format*, not Pointcept.

Pointcept is a CVPR'24 Oral research project from MMLab/Meta AI,
published under the official `Pointcept` HF org — as trusted a source
as research checkpoints get. The detected pickle imports in this
checkpoint are all standard torch / numpy / collections globals
(no red flags — verified at download time).

We additionally load with `torch.load(..., weights_only=True)` in both
`ptv3.py` and `randla.py`, which blocks pickle code-execution at load
time regardless of file contents. There's a fallback to
`weights_only=False` with a warning if the checkpoint contains globals
not on torch's allowlist (the PTv3 checkpoint includes an OneCycleLR
scheduler state which fails strict mode before torch 2.6) — in that
case the fallback is only safe because the file came from the trusted
HF mirror.

### Run

```bash
docker compose -f docker-compose.ml.yml up -d
docker compose -f docker-compose.ml.yml logs -f cloud2bim-ml
```

Web UI: `http://<host>:8001`

In the wizard:
1. Step 1 → pick the scan from the file browser. Anything mounted under
   `/drives/<Namn>` is listed automatically — `vasakronan temp.e57`
   under `Punktmoln` with the default compose layout.
2. Step 2 → **Pipeline-läge** = "Hybrid" (or "ML-only" once trusted)
3. Step 2 → **Backend** = PointTransformer V3
4. Step 2 → **Has RGB** = Auto

The first run downloads PTv3 weights into the `cloud2bim-models` named
volume (~160 MB). The volume persists across container rebuilds so the
download happens once per host.

### Quick GPU sanity test inside the container

```bash
docker compose -f docker-compose.ml.yml exec cloud2bim-ml python -c "
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name() if torch.cuda.is_available() else '')
from pointcept.models.point_transformer_v3 import PointTransformerV3
print('PTv3 import OK')
"
```

Expected:
```
cuda: True NVIDIA RTX ...
PTv3 import OK
```

If `cuda: False` — `nvidia-container-toolkit` can't see the GPU. Verify
the toolkit is installed and configured for Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that command also fails, follow the NVIDIA container toolkit install
guide for Ubuntu 24 — typically `sudo apt-get install -y nvidia-container-toolkit`
followed by `sudo nvidia-ctk runtime configure --runtime=docker` and a
`sudo systemctl restart docker`.

---

## Local pip workflow (alternative)

## 1. Install ML dependencies (Windows + CUDA)

Pointcept's PointTransformer V3 needs torch + spconv built against your
CUDA version. The team's GPU runs CUDA 11.8, so use the matching wheels:

```powershell
# Inside the project venv (PyCharm's Python interpreter, or a fresh one):
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

`requirements-ml.txt` installs `torch==2.1.0+cu118`, `spconv-cu118`, and
Pointcept from git. Add `tensorboard` if you want training metrics later.

Sanity check:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
python -c "from pointcept.models.point_transformer_v3 import PointTransformerV3; print('PTv3 import OK')"
```

Expected output:

```
cuda: True NVIDIA RTX ...
PTv3 import OK
```

## 2. Trigger weights download (or place manually)

First inference call auto-downloads to
`%LOCALAPPDATA%\cloud2bim\models\ptv3-s3dis-area5.pth` (~160 MB).
Override the cache dir with `CLOUD2BIM_MODELS_DIR`.

If the auto-download URL is unreachable (corporate firewall etc.), grab
the checkpoint manually from the Pointcept Hugging Face mirror and drop
it at the path above.

## 3. Edit the sample config

Open `configs/v3-sample-hybrid.yaml`. Edit three paths:

```yaml
io:
  input_files:
    - "\\\\192.168.2.79\\ssd250\\scan2bim\\pointclouds\\vasakronan temp.e57"
  output_ifc: "C:/Users/jacob_000/Desktop/v3-smoke/output.ifc"
  work_dir:   "C:/Users/jacob_000/Desktop/v3-smoke"
```

(Note Windows path doubling for YAML.)

For the very first run, lower the cost by setting:

```yaml
io:
  dilution_factor: 50      # ~5M points instead of ~250M
```

Once you've confirmed it works, drop dilution back to 10 or off entirely.

## 4. Run

```powershell
python -m cloud2bim run configs/v3-sample-hybrid.yaml
```

Watch the log for these key lines (order matters):

```
INFO  cloud2bim.pipeline: Loaded N points (rgb=True)
INFO  cloud2bim.segmentation.factory: ...                              # picks PTv3
INFO  cloud2bim.segmentation.weights: Downloading ptv3-s3dis-area5 ... # first run only
INFO  cloud2bim.segmentation.ptv3: PTv3 inference on N points (rgb=True)
INFO  cloud2bim.segmentation.ptv3: PTv3 chunking: ... → K-cell ...m grid
INFO  cloud2bim.segmentation.ptv3: PTv3 done — class breakdown: wall=..., floor=..., ceiling=..., ...
INFO  cloud2bim.pipeline: ─── Slab segmentation (mode=hybrid, fallback algo=v1) ───
INFO  cloud2bim.extraction.slabs_ml: ML slabs: ... floor points, ... ceiling points
INFO  cloud2bim.extraction.slabs_ml: ML slabs: M floor surfaces + M ceiling surfaces
INFO  cloud2bim.pipeline: Slabs: ... in ...s
INFO  cloud2bim.pipeline: ─── Wall & opening segmentation ───
INFO  cloud2bim.extraction.walls_ml: ML walls storey 0: N wall points (...% of storey)
INFO  cloud2bim.extraction.walls_ml: ML walls storey 0: K DBSCAN clusters
INFO  cloud2bim.extraction.walls_ml: ML walls storey 0: M walls finalised
... (repeated per storey) ...
INFO  cloud2bim.pipeline: ─── DONE in ...s: X slabs, Y walls, Z openings, ... ───
```

If hybrid falls back, you'll also see:

```
WARNING cloud2bim.pipeline: Hybrid: ML found 0 slabs (<2) — falling back to geometric (v1)
INFO    cloud2bim.pipeline: Hybrid storey 0: only 50 wall-labelled points (<5000 threshold) — using geometric
```

That's not a failure — it's the safety net doing its job.

## 5. Inspect the result

```powershell
# Open the IFC in Revit / BIMcollab Zoom / IFC++ / xeokit viewer
explorer C:\Users\jacob_000\Desktop\v3-smoke
```

The work dir also contains:

- `points.npz`     — XYZ (+ RGB) after prepare
- `labels.npy`     — per-point class ids (re-used between runs)
- `slabs.pkl`, `walls.pkl`, `openings.pkl` — stage outputs
- `output_preview.png` — floor-plan PNG

To re-run with a different `pipeline_mode` without re-downloading 250M
points: edit the YAML and run again. `labels.npy` is cached unless the
point count changes.

## 6. Run from the web wizard instead of the CLI

```powershell
docker compose up --build   # builds the python:3.11 image
# Then open http://localhost:8001 and pick:
#   Step 1: "Network path" → \\192.168.2.79\ssd250\scan2bim\pointclouds\vasakronan temp.e57
#   Step 2: Pipeline-läge = "Hybrid"
#           ML backend = PTv3
#           Has RGB = Auto
```

⚠️ The default Dockerfile is CPU-only Python — ML inference will be slow
(minutes per storey on CPU). For GPU inside Docker you need
`nvidia-docker` + a CUDA-base image; not set up in this repo yet. Run
the CLI on the host instead.

## 7. What "good" looks like

For an indoor office scan like vasakronan:

- **Class breakdown** should be dominated by `wall`, `floor`, `ceiling`
  and `clutter` (chairs/tables/sofas). If everything is `clutter` the
  RGB normalisation or weight loading is off — check the breakdown log
  line.
- **ML slabs** should match real floor levels within ±5 cm.
- **ML walls per storey** should be roughly comparable to the geometric
  count, but with fewer phantom walls in heavily furnished rooms.
- **Openings** caught with windows/doors visible in the scan.

If hybrid falls back to geometric on every storey, the ML stack ran but
returned too few wall points — most likely cause is wrong RGB scaling
(check the breakdown — `clutter` dominating points to feature norm being
off) or a checkpoint that doesn't match `PTV3_IN_CHANNELS=6`.
