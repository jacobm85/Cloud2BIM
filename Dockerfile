FROM python:3.11

# System libraries for Open3D, OpenCV headless, matplotlib, and download tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in a separate layer so rebuilds are fast
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application source
COPY . .

# Pre-create directories the pipeline writes to at runtime
RUN mkdir -p web/uploads web/jobs images/pdf images/wall_outputs_images

# ── Download xeokit-bim-viewer (includes its own tested web-ifc) ──────────
# The full viewer package solves web-ifc version compatibility issues because
# it ships its own libs/web-ifc/ at a version it was tested against.
RUN curl -fsSL \
    "https://github.com/xeokit/xeokit-bim-viewer/archive/refs/heads/master.zip" \
    -o /tmp/bimviewer.zip && \
    unzip -q /tmp/bimviewer.zip -d /tmp && \
    mkdir -p web/static/bimviewer && \
    cp -r /tmp/xeokit-bim-viewer-master/dist   web/static/bimviewer/dist && \
    cp -r /tmp/xeokit-bim-viewer-master/src    web/static/bimviewer/src && \
    cp -r /tmp/xeokit-bim-viewer-master/libs   web/static/bimviewer/libs && \
    rm -rf /tmp/bimviewer.zip /tmp/xeokit-bim-viewer-master

EXPOSE 8000

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
