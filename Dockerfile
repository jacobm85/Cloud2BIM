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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in a separate layer so rebuilds are fast
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application source
COPY . .

# Capture git commit info at build time so the GUI version pill shows
# the actually-running build. Pass via build-args; the wrapper script
# in scripts/build.sh fills them from `git rev-parse` on the host.
# (.git is excluded by .dockerignore so we can't shell-out to git here.)
ARG GIT_SHA=dev
ARG GIT_DATE=
ARG GIT_BRANCH=
# Fall back to current build time when GIT_DATE wasn't passed (typically
# when running `docker compose up --build` directly instead of through
# scripts/build.sh). The GUI version pill needs *some* date string to
# show; the build timestamp is at least informative.
RUN if [ -n "$GIT_DATE" ]; then \
        printf '%s %s %s\n' "$GIT_SHA" "$GIT_DATE" "$GIT_BRANCH" > /app/VERSION; \
    else \
        printf '%s %s %s\n' "$GIT_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_BRANCH" > /app/VERSION; \
    fi

# Pre-create directories the pipeline writes to at runtime
RUN mkdir -p web/uploads web/jobs images/pdf images/wall_outputs_images

# ── IFC viewer assets (downloaded from npm via jsDelivr) ─────────────────
# web-ifc@0.0.44: UMD bundle that sets window.WebIFC when loaded as <script>
# xeokit-sdk: ES module IFC viewer library, tested with web-ifc@0.0.44
RUN mkdir -p web/static/bimviewer/libs/web-ifc web/static/bimviewer/dist && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/web-ifc@0.0.44/web-ifc-api.js" \
         -o web/static/bimviewer/libs/web-ifc/web-ifc-api.js && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/web-ifc@0.0.44/web-ifc.wasm" \
         -o web/static/bimviewer/libs/web-ifc/web-ifc.wasm && \
    curl -fsSL "https://cdn.jsdelivr.net/npm/@xeokit/xeokit-sdk/dist/xeokit-sdk.es.js" \
         -o web/static/bimviewer/dist/xeokit-sdk.es.js

EXPOSE 8001

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
