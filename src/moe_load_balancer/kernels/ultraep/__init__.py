"""Placement and rerouting kernels copied from UltraEP v1.0.0.

This package contains no UltraEP communication runtime, and MLB takes no
dependency on the ``ultra_ep`` package: these kernels only compute. Making
memory match a placement they solved is the serving framework's job, done with
the mover it already has, driven by the before/after maps the solve returns.
"""

from .backend import UltraEPPlacementBackend
from .config import UltraEPPlacementTuning, load_placement_tuning_from_env

__all__ = [
    "UltraEPPlacementBackend",
    "UltraEPPlacementTuning",
    "load_placement_tuning_from_env",
]
