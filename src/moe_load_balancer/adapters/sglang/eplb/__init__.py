"""SGLang adapter for the EPLB L1 planner.

Exposes:

* :class:`SGLangEPLBRuntime` -- the entry point SGLang's
  ``ExpertLocationMetadata.init_by_eplb`` redirects through. Also offers a
  ``plan_placement(server_args, model_config, collector)`` method for the
  periodic-rebalance loop, which pulls stats via the framework-neutral
  :class:`StatsCollector` protocol.

* :class:`SGLangStatsCollector` -- adapter from
  ``ExpertDistributionRecorder`` to the framework-neutral
  :class:`StatsCollector` protocol. Natural pairing with
  ``SGLangEPLBRuntime.plan_placement``.

Importing this module is cheap (no sglang import at module load); the
sglang dependency is deferred to call time.
"""

from __future__ import annotations

from moe_load_balancer.adapters.sglang.eplb.runtime import SGLangEPLBRuntime
from moe_load_balancer.adapters.sglang.eplb.stats import SGLangStatsCollector

__all__ = ["SGLangEPLBRuntime", "SGLangStatsCollector"]
