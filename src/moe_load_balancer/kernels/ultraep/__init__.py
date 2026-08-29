"""Placement and rerouting kernels copied from UltraEP v1.0.0.

This package deliberately contains no UltraEP communication runtime.  MLB owns
the placement algorithm; serving frameworks call the standalone ``ultra_ep``
package when they want to transfer expert weights.
"""

from .backend import UltraEPPlacementBackend
from .config import UltraEPPlacementTuning, load_placement_tuning_from_env

__all__ = [
    "UltraEPPlacementBackend",
    "UltraEPPlacementTuning",
    "load_placement_tuning_from_env",
]
