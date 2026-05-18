# Build and optionally start Cloud2BIM containers with the current
# git commit info baked in. Mirror of scripts/build.sh for Windows.
#
# Usage:
#   scripts\build.ps1               # docker compose build
#   scripts\build.ps1 up            # docker compose up
#   scripts\build.ps1 -f docker-compose.ml.yml up --build
$ErrorActionPreference = "Stop"

$repoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) { $repoRoot = (Get-Location).Path }
Set-Location $repoRoot

$env:GIT_SHA = (git rev-parse --short HEAD 2>$null)
if (-not $env:GIT_SHA) { $env:GIT_SHA = "dev" }
$env:GIT_DATE = (git log -1 --format=%cI 2>$null)
if (-not $env:GIT_DATE) { $env:GIT_DATE = "" }
$env:GIT_BRANCH = (git rev-parse --abbrev-ref HEAD 2>$null)
if (-not $env:GIT_BRANCH) { $env:GIT_BRANCH = "" }

Write-Host "Building with GIT_SHA=$env:GIT_SHA GIT_DATE=$env:GIT_DATE GIT_BRANCH=$env:GIT_BRANCH"

if ($args.Count -eq 0) {
  docker compose build
} else {
  docker compose @args
}
