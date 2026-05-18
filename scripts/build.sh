#!/usr/bin/env bash
# Build and (optionally) start the Cloud2BIM containers with the current
# git commit info baked in as build-args. The Dockerfile writes them to
# /app/VERSION so the GUI version pill always shows the running build.
#
# Usage:
#   scripts/build.sh                  # builds the base image
#   scripts/build.sh up               # builds and `up -d` the base image
#   scripts/build.sh -f docker-compose.ml.yml up   # same for the ML image
#
# All arguments after the first are forwarded to `docker compose`.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

export GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
export GIT_DATE="$(git log -1 --format=%cI 2>/dev/null || echo '')"
export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"

echo "Building with GIT_SHA=$GIT_SHA GIT_DATE=$GIT_DATE GIT_BRANCH=$GIT_BRANCH"

if [[ $# -eq 0 ]]; then
  exec docker compose build
fi

# Forward `up`, `up -d`, `-f docker-compose.ml.yml up --build` etc.
exec docker compose "$@"
