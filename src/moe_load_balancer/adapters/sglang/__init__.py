"""SGLang adapter package.

Top-level convenience exports are limited to symbols that do **not** require
``torch`` / ``triton`` to import. Callers that need GPU runtime helpers
(``count_local_routed_per_rank``, ``WaterfillRuntime``) must import them from
the explicit submodule path so dev environments without ``triton`` can still
import this package and run the framework-neutral contract tests.

- ``SGLangPlacementAdapter`` (this re-export): pure CPU, no triton.
- ``moe_load_balancer.adapters.sglang.load_kernels``: needs torch + triton.
- ``moe_load_balancer.adapters.sglang.waterfill``: needs torch + triton + sglang.
"""

from moe_load_balancer.adapters.sglang.providers import SGLangPlacementAdapter

__all__ = [
    "SGLangPlacementAdapter",
]
