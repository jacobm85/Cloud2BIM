# Cloud2BIM v3-ml — GPU image with PointTransformer V3 + RandLA-Net.
#
# Targets a CUDA 12.x host (server .74 runs CUDA 12.5 driver on
# Ubuntu 24 LTS). nvidia-container-toolkit ≥ 1.16 required on the host;
# the container itself uses the cu124 runtime, which is forward-binary-
# compatible with 12.5 drivers.
#
# Build:   docker compose -f docker-compose.ml.yml build
# Run:     docker compose -f docker-compose.ml.yml up

# pytorch/pytorch already ships torch 2.5.1+cu124 + cudnn 9 — saves us
# from compiling CUDA bindings or pulling a 4 GB torch wheel manually.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System libs: open3d, opencv-headless, matplotlib, plus git+curl for
# the Pointcept pip install and the viewer asset download.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Non-ML deps first so an ML bump doesn't bust the whole cache.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# ML deps. spconv-cu120 is forward-compatible with cu124+ drivers
# (NVIDIA ABI is stable within the 12.x family).
#
# Pointcept's repo ships no setup.py / pyproject.toml so pip install
# from git fails ("not a Python project"). Clone it into a known path
# and add to PYTHONPATH instead — PTv3 is a self-contained module that
# only imports torch + spconv + a few utility libs from elsewhere.
# Pinned to a known-working commit so main-branch churn doesn't break
# rebuilds; bump deliberately when you want a newer PTv3.
RUN pip install --no-cache-dir \
    spconv-cu120==2.3.6 \
    einops \
    addict \
    timm

# Pointcept's models/__init__.py loads default.py at import time, which
# imports the full PyG extension suite (torch_scatter, torch_cluster,
# torch_sparse, torch_spline_conv). PTv3 itself doesn't use them, but
# Python loads the whole package when we import PTv3 from it. Install
# all four prebuilt PyG wheels in one go — building any of them from
# source against torch+CUDA takes 10+ min and often fails on ABI drift.
#
# torch-2.5.0+cu124 wheels are binary-compatible with our 2.5.1+cu124
# base image (PyG hasn't shipped a 2.5.1 index yet).
RUN pip install --no-cache-dir \
    torch_scatter \
    torch_cluster \
    torch_sparse \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

RUN git clone https://github.com/Pointcept/Pointcept.git /opt/pointcept_src && \
    cd /opt/pointcept_src && \
    git checkout d74c646db6abec569d0f23e0c34e7ddfce142789 && \
    rm -rf .git
ENV PYTHONPATH=/opt/pointcept_src:${PYTHONPATH}

# Application source
COPY . .

# Runtime dirs the pipeline writes to (mounted via compose volumes)
RUN mkdir -p web/uploads web/jobs /models /data

# IFC viewer assets — identical to the base Dockerfile
RUN mkdir -p web/static/bimviewer/libs/web-ifc web/static/bimviewer/dist && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/web-ifc@0.0.44/web-ifc-api.js" \
         -o web/static/bimviewer/libs/web-ifc/web-ifc-api.js && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/web-ifc@0.0.44/web-ifc.wasm" \
         -o web/static/bimviewer/libs/web-ifc/web-ifc.wasm && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/@xeokit/xeokit-sdk/dist/xeokit-sdk.es.js" \
         -o web/static/bimviewer/dist/xeokit-sdk.es.js

# Persistent model cache lives at /models (compose-mounted volume),
# so weights survive container rebuilds.
ENV CLOUD2BIM_MODELS_DIR=/models

EXPOSE 8001

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
