# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SGLang-backed :class:`LPLBKernels` implementation.

Thin delegation layer over ``sglang.jit_kernel.lplb.cuda_solver``. Constructed
once by :class:`LPLBRuntime` and threaded through to :class:`LPLBL2Router`
via constructor injection. Sets the "Option 1" path of the open
kernel-adapter decision (see
``load_balancer_project/DECISION_lplb_kernel_adapter.md``); a future TRT-LLM
or vendored implementation supplies its own :class:`LPLBKernels` without
changing the policy or the core contract.
"""

from __future__ import annotations

from typing import Any, Optional


class SGLangLPLBKernels:
    """Adapter that satisfies :class:`moe_load_balancer.core.l2.LPLBKernels`.

    The constructor imports ``sglang.jit_kernel.lplb.cuda_solver`` lazily so
    that pure-Python contract tests (which don't have sglang installed)
    can still import this module. Once constructed, every call delegates
    directly into the SGLang kernel module — no extra Python work happens
    on the per-batch hot path beyond the protocol method dispatch.
    """

    def __init__(self) -> None:
        try:
            from sglang.jit_kernel.lplb import cuda_solver
        except ImportError as exc:  # pragma: no cover — exercised only in non-SGLang envs
            raise ImportError(
                "SGLangLPLBKernels requires sglang.jit_kernel.lplb.cuda_solver, "
                "which is part of an SGLang build that ships the LPLB CUDA "
                "kernels (see sgl-project/sglang#24515). Install sglang in the "
                "active environment, or pick a different LPLBKernels "
                "implementation."
            ) from exc

        self._cuda = cuda_solver

    def warmup(self, nc: int, nv: int, num_iters: int, device: Any) -> None:
        self._cuda.warmup(nc=nc, nv=nv, num_iters=num_iters, device=device)

    def prep_lp_inputs(
        self,
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
    ) -> None:
        self._cuda.prep_lp_inputs(
            A_full,
            b,
            t1,
            global_counts,
            log_single,
            log_replicated,
            B1,
            A_base_row_sum,
        )

    def solve_ipm(self, A, b, c, num_iters: int = 5, result: Optional[Any] = None):
        return self._cuda.solve_ipm(A, b, c, num_iters=num_iters, result=result)

    def extract_log2phy_prob(
        self, log2phy_prob, x, t1, phy_single, phy_replicated, log2phy
    ) -> None:
        self._cuda.extract_log2phy_prob(
            log2phy_prob, x, t1, phy_single, phy_replicated, log2phy
        )

    def dispatch_probability(self, topk_ids, log2phy_prob, log2phy_map):
        return self._cuda.dispatch_probability(topk_ids, log2phy_prob, log2phy_map)
