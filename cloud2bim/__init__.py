"""Cloud2BIM — Scan-to-BIM pipeline.

Production rewrite focused on robustness against real-world point clouds
(furniture, noise, large absolute coordinates) by combining ML semantic
segmentation with histogram-based geometric extraction.
"""

__version__ = "2.0.0-dev"


def git_revision() -> str:
    """Short git hash of the running code, or 'unknown'.

    Logged at every pipeline/stage start so a field report can always
    be matched to the exact code that produced it — a silent version
    mismatch once burned a full day of field testing.
    """
    import subprocess
    from pathlib import Path
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"
