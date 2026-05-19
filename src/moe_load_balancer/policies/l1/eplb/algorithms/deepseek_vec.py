"""Stub for the GPU-vectorized DeepSeek EPLB variant.

To enable, vendor ``python/sglang/srt/eplb/eplb_algorithms/deepseek_vec.py``
from SGLang into this file. The dispatcher in
:mod:`moe_load_balancer.policies.l1.eplb.algorithms` already wires the
``EplbAlgorithm.deepseek_vec`` / ``deepseek_vec_hierarchical`` enum entries
to ``rebalance_experts`` here, so vendoring the file is sufficient — no
dispatcher edit is needed.

Until then, calling these variants raises ``NotImplementedError`` to fail
loudly rather than silently downgrading to ``deepseek``.
"""

from __future__ import annotations


def rebalance_experts(*args, **kwargs):  # noqa: D401 — stub
    raise NotImplementedError(
        "deepseek_vec is a stub in this POC; vendor "
        "sglang/python/sglang/srt/eplb/eplb_algorithms/deepseek_vec.py to enable. "
        "The baseline benchmark (--eplb-algorithm auto on DeepSeek-V3) resolves to "
        "deepseek_hierarchical and does not need this variant."
    )
