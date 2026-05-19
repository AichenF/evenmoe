"""Stub for the Elastic-EP-aware DeepSeek EPLB variant.

To enable, vendor ``python/sglang/srt/eplb/eplb_algorithms/elasticity_aware.py``
from SGLang into this file. The upstream file imports
``rebalance_experts_hierarchical`` from ``deepseek``; when vendoring, change
that import to ``from moe_load_balancer.policies.l1.eplb.algorithms.deepseek
import rebalance_experts_hierarchical``.

The dispatcher already wires ``EplbAlgorithm.elasticity_aware*`` enum entries
to ``rebalance_experts`` here and threads through the ``active_ranks``
argument; the SGLang adapter is the one that pulls ``active_ranks`` from
``ElasticEPStateManager`` so this module stays framework-neutral.
"""

from __future__ import annotations


def rebalance_experts(*args, **kwargs):  # noqa: D401 — stub
    raise NotImplementedError(
        "elasticity_aware is a stub in this POC; vendor "
        "sglang/python/sglang/srt/eplb/eplb_algorithms/elasticity_aware.py to enable. "
        "The baseline benchmark does not exercise Elastic-EP."
    )
