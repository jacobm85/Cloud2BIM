"""V1 (original Cloud2BIM) detection, ported from the 20fa12e state of master.

Self-contained ports of the wall, slab and opening algorithms in:
    https://github.com/VaclavNezerka/Cloud2BIM

20fa12e adds two local patches on top of upstream — PCA rotation in
walls and a 0.4 (vs 0.6) slab density threshold — that the user has
empirically validated as producing better results on real scans. Both
are preserved here.

Kept as the known-good baseline so we can A/B against v2's variant and
roll back changes that don't measurably improve detection. Don't
refactor without understanding the original — the goal is behavioural
parity with 20fa12e, not clever code.

Selecting algorithm="v1" in the wizard routes ALL three stages (slabs,
walls, openings) to these ports, giving a self-contained v1 pipeline
that's independent of the v2/v3 code paths.
"""
from cloud2bim.legacy.openings_v1 import detect_openings_v1
from cloud2bim.legacy.slabs_v1 import detect_slabs_v1
from cloud2bim.legacy.walls_v1 import detect_walls_v1

__all__ = ["detect_walls_v1", "detect_slabs_v1", "detect_openings_v1"]
