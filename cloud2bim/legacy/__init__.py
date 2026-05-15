"""V1 (original Cloud2BIM) wall + slab detection, ported from master.

Self-contained ports of the algorithms in:
    https://github.com/VaclavNezerka/Cloud2BIM

Kept here as the known-good baseline so we can A/B against v2's variant
and roll back changes that don't measurably improve detection.

Don't refactor anything in this package without understanding the
original — the goal is bit-for-bit behavioural parity with master, not
clever code.
"""
from cloud2bim.legacy.slabs_v1 import detect_slabs_v1
from cloud2bim.legacy.walls_v1 import detect_walls_v1

__all__ = ["detect_walls_v1", "detect_slabs_v1"]
