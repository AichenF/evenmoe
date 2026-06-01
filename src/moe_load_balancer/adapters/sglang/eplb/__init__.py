"""SGLang adapter for the EPLB L1 planner.

Exposes:

* :class:`SGLangEPLBRuntime` -- stateful facade that owns the
  framework-neutral pieces (recorder, provider, policy) and routes
  sglang's ``ExpertLocationMetadata.init_by_eplb`` static call to MLB.

* :class:`SGLangStatsCollector` -- adapter from
  ``ExpertDistributionRecorder.dump_record`` to the framework-neutral
  :class:`StatsCollector` protocol. Used when L1 wants to read sglang's
  *existing* recorder state (read-only view). The newer
  :class:`SGLangStatsBridge` is preferred when MLB owns the buffer.

* :class:`SGLangStatsBridge` -- the proxy that replaces sglang's
  ``_StatAccumulator._Buffer``. SGLang pushes per-forward observations
  into MLB through this bridge, and pulls the aggregated dict back out
  through it at rebalance time.

* :func:`get_default_runtime` / :func:`set_default_runtime` --
  module-level slot for the runtime instance sglang's static-method
  shim picks up.

Importing this module is cheap (no sglang import at module load); the
sglang dependency is deferred to call time.
"""

from __future__ import annotations

from moe_load_balancer.adapters.sglang.eplb.runtime import (
    SGLangEPLBRuntime,
    clear_default_runtime,
    ensure_default_runtime,
    get_default_runtime,
    set_default_runtime,
)
from moe_load_balancer.adapters.sglang.eplb.stats import SGLangStatsCollector
from moe_load_balancer.adapters.sglang.eplb.stats_bridge import SGLangStatsBridge

__all__ = [
    "SGLangEPLBRuntime",
    "SGLangStatsBridge",
    "SGLangStatsCollector",
    "clear_default_runtime",
    "ensure_default_runtime",
    "get_default_runtime",
    "set_default_runtime",
]
