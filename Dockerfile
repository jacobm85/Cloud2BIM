FROM python:3.11

# System libraries required by Open3D, OpenCV headless, and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in a separate layer so rebuilds are fast
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application source
COPY . .

# Pre-create directories the pipeline writes to at runtime
RUN mkdir -p web/uploads web/jobs images/pdf images/wall_outputs_images

# Download IFC viewer assets at build time — served locally, no CDN at runtime
RUN mkdir -p web/static/vendor && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/@xeokit/xeokit-sdk/dist/xeokit-sdk.es.js" \
         -o web/static/vendor/xeokit-sdk.es.js && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/web-ifc@0.0.57/web-ifc-api.js" \
         -o web/static/vendor/web-ifc-api.js && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/web-ifc@0.0.57/web-ifc.wasm" \
         -o web/static/vendor/web-ifc.wasm

EXPOSE 8000

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
