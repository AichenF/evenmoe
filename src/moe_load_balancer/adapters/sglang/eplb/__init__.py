"""SGLang adapter for the EPLB L1 planner.

Exposes :class:`SGLangEPLBRuntime`, the single entry point SGLang's
``ExpertLocationMetadata.init_by_eplb`` redirects through. Importing this
module is cheap (no sglang import at module load); the sglang dependency is
deferred to the ``init_by_eplb`` call site.
"""

from __future__ import annotations

from moe_load_balancer.adapters.sglang.eplb.runtime import SGLangEPLBRuntime

__all__ = ["SGLangEPLBRuntime"]
