# SGLang Waterfill Adapter Restructure — Design

**Date:** 2026-05-09
**Scope:** `moe_load_balancer` repo + SGLang worktree `sglang-main-moe-lb-waterfill`
**Goal:** Rebuild PR #19290 (DeepEP Waterfill) on top of `moe_load_balancer` as a properly layered module, while keeping SGLang glue minimal and aligning the layout with `docs/architecture.md` Stage 2 contracts.

This document is the canonical record of the architecture decisions for this refactor. It supersedes the previous skeleton-style adapter implementation.

## 1. Motivation

The pre-restructure state had three issues:

1. **Algorithm duplication.** `policies/l2/waterfill.py` held a 391-line CPU reference (`SharedExpertWaterfillPolicy`); the GPU implementation lived in `adapters/sglang/deepep_waterfill.py` with a separate Triton kernel. The same algorithm was specified twice and could drift.
2. **Adapter mixing concerns.** The SGLang adapter folder bundled together (a) framework-neutral algorithm scaffolding, (b) GPU runtime, (c) SGLang-specific glue, with file names (`waterfill.py`, `deepep_waterfill.py`) that did not communicate the split.
3. **Schema sloppiness.** `RoutingRequest.current_rank_load` had mode-dependent semantics (local routed counts in static, effective load in dynamic). `RoutingPolicyConfig.metadata` was being used as a per-step data channel rather than per-policy stable config. `SGLangRoutingContext` duplicated `RoutingRequest` fields.

Per `docs/architecture.md`, Stage 2 commits to a `core` / `policies` / `adapters` separation with canonical `TokenRouter` / `RoutingRequest` / `RoutingDecision` / `RoutingLoadHistoryProvider` contracts. The refactor brings the implementation in line with that commitment.

## 2. Design Principles

1. **`core/` is contract-only.** Types and protocols. No GPU code, no framework imports.
2. **`policies/l2/<algo>.py` is the algorithm.** A single GPU implementation per algorithm, implementing `core.TokenRouter`. No CPU reference duplication.
3. **`adapters/<framework>/` is glue.** Extracts framework state into canonical types, hosts framework-specific kernels (materialization, framework-coupled load extraction), executes the runtime pipeline.
4. **`metadata` is per-policy stable config.** Tensors flow through canonical `RoutingRequest` fields or `RoutingLoadHistoryProvider`, never through `metadata`.
5. **Asymmetry tolerated when justified by performance.** PR #19290's static-mode `target_total = 0` sentinel + kernel-derive optimization is preserved (avoids an additional host-GPU sync per layer in static mode).

## 3. Final Module Layout

```
moe_load_balancer/src/moe_load_balancer/
├── core/                                # Stage 2 contract layer (preserved)
│   ├── types.py                         # RoutingRequest schema updated, RoutingLoadHistory rename
│   ├── l1.py                            # unchanged
│   └── l2.py                            # RoutingLoadHistoryProvider rename
│
├── policies/
│   └── l2/
│       └── waterfill.py    (~250L)      # WaterfillL2Router(TokenRouter) + Triton select kernel
│
└── adapters/sglang/
    ├── __init__.py                      # back-compat re-exports for tests
    ├── providers.py        (~22L)       # SGLangPlacementAdapter (L1 generic, shared across L2)
    ├── load_kernels.py     (~70L)       # count_local_routed_per_rank (DeepEP layout assumption)
    └── waterfill/                       # L2 Waterfill-specific subpackage
        ├── __init__.py                  # exports WaterfillRuntime
        └── runtime.py      (~280L)      # WaterfillRuntime + materialize kernel + EP all-reduce
```

## 4. core/ Changes

### 4.1 `core/types.py`

```python
@dataclass(frozen=True, kw_only=True)
class RoutingRequest:
    layer_id: int
    logical_topk_ids: TensorLike
    topk_weights: TensorLike
    placement: Optional[PlacementSnapshot] = None         # NEW: Optional (Waterfill skips)
    stage: Optional[RoutingStage] = None
    routed_rank_load: Optional[TensorLike] = None         # NEW: per-rank routed counts (post-collective)
    active_rank_token_count: Optional[TensorLike] = None  # NEW: per-rank active tokens (post-collective)
    token_count: Optional[TensorLike] = None
    lb_token_mask: Optional[TensorLike] = None
    routed_scaling_factor: float = 1.0
    policy: RoutingPolicyConfig | str = ...
    metadata: Mapping[str, Any] = ...
    # REMOVED: current_rank_load (mode-dependent semantics, replaced by routed + active)


@dataclass(frozen=True)
class RoutingPolicyConfig:
    name: str = DISABLED                                  # plain string, no enum constraint
    enabled_stages: Optional[frozenset[RoutingStage]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # docstring约定: metadata只装 per-policy 稳定 config (immutable scalars).
    #   Tensors must NOT go here; use RoutingRequest top-level fields or
    #   RoutingLoadHistoryProvider for tensor inputs.


@dataclass(frozen=True)
class RoutingLoadHistory:                                 # was RoutingLoadView
    live_rank_load: Optional[TensorLike] = None
    historical_rank_load: Optional[TensorLike] = None              # was prior_rank_load
    historical_expert_load: Optional[TensorLike] = None            # was physical_expert_load
    historical_rank_expert_load: Optional[TensorLike] = None       # was rank_expert_load
    metadata: Mapping[str, Any] = ...
```

### 4.2 `core/l2.py`

```python
@runtime_checkable
class RoutingLoadHistoryProvider(Protocol):               # was RoutingLoadProvider
    def load_history(self, layer_id: int) -> RoutingLoadHistory:   # was load_view
        ...

class TokenRouter(Protocol):                              # unchanged
    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        ...
```

### 4.3 Field semantics

| Field | Purpose | Set by adapter when |
|---|---|---|
| `routed_rank_load` | Per-rank routed expert assignment count | Always (static: local_routed; dynamic: post-all-reduce global routed) |
| `active_rank_token_count` | Per-rank active token count | Only in dynamic mode (post-all-reduce) |

Static / dynamic mode is signaled by **whether `active_rank_token_count` is set**, not by an explicit mode flag.

## 5. policies/l2/waterfill.py

```python
import torch, triton, triton.language as tl
from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import RoutingRequest, RoutingDecision

class WaterfillL2Router(TokenRouter):
    """SGLang-compatible shared-expert Waterfill L2 router.

    Implements core.TokenRouter via the standard pipeline:
        RoutingRequest -> route_tokens -> RoutingDecision

    Waterfill is a degenerate L2 case:
    - Does not need logical->physical translation (adapter provides physical IDs already).
    - Only outputs shared_expert_rank; routed top-k passes through unchanged.

    Static / dynamic mode is auto-distinguished by presence of
    request.active_rank_token_count.
    """

    def __init__(self, *, source_rank: int, local_preference: float, world_size: int):
        self.source_rank = source_rank
        self.local_preference = local_preference
        self.world_size = world_size

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        # Stage gating (core protocol path)
        if not request.policy.should_run_for_stage(request.stage):
            return RoutingDecision(
                routed_physical_topk_ids=request.logical_topk_ids,
                topk_weights=request.topk_weights,
                shared_expert_rank=None,
            )

        topk_ids = request.logical_topk_ids
        topk_size = topk_ids.shape[1]
        routed = request.routed_rank_load
        active = request.active_rank_token_count
        experts_per_rank = int(request.policy.metadata["experts_per_rank"])

        if routed is None:
            raise ValueError("Waterfill needs routed_rank_load.")

        if active is None:
            # Static mode: rank_load = A only, target_total=0 sentinel for kernel derive
            rank_load = routed
            target_total = 0
            allow_all_ranks = True
        else:
            # Dynamic mode: rank_load = A + B, explicit target_total
            rank_load = routed + active
            total_routed = routed.sum()
            total_tokens = total_routed // topk_size
            total_effective = rank_load.sum()
            target_total = int(
                (total_effective + total_tokens + self.world_size - 1) // self.world_size
            )
            allow_all_ranks = bool(rank_load.max() <= target_total)

        chosen_rank = _waterfill_select_via_kernel(
            rank_load=rank_load,
            topk_ids=topk_ids,
            target_total=target_total,
            allow_all_ranks=allow_all_ranks,
            source_rank=self.source_rank,
            local_preference=self.local_preference,
            world_size=self.world_size,
            experts_per_rank=experts_per_rank,
        )

        return RoutingDecision(
            routed_physical_topk_ids=topk_ids,
            topk_weights=request.topk_weights,
            shared_expert_rank=chosen_rank,
        )


@triton.jit
def _waterfill_select_kernel(...): ...
def _waterfill_select_via_kernel(...): ...
```

The Triton kernel keeps PR #19290's two paths internally: when `target_total == 0`, it derives via `(sum(rank_load) + sum(rank_load)/topk + ws - 1) / ws` (correct in static where `rank_load == A`); otherwise it uses the precomputed value (required in dynamic since `rank_load == A + B` makes naive derivation incorrect).

## 6. adapters/sglang/

### 6.1 `providers.py`

```python
"""L1 SGLang adapters (generic across L2 algorithms)."""

class SGLangPlacementAdapter(PlacementProvider):
    """Implements core.l1.PlacementProvider; converts SGLang ExpertLocationMetadata
    into framework-neutral PlacementSnapshot."""
    def snapshot(self, layer_id: int) -> PlacementSnapshot: ...
```

Future LPLB-style policies that need a `RoutingLoadHistoryProvider` implementer add it to this file (or a sibling).

### 6.2 `load_kernels.py`

```python
"""SGLang/DeepEP-specific GPU helpers for L2 load extraction.

Assumes contiguous expert-to-rank layout (rank = expert_id // experts_per_rank).
This is the SGLang/DeepEP/vLLM/TRT-LLM convention. Other adapters using
different layouts must provide their own load extraction kernels.
"""

@triton.jit
def _count_routed_per_rank_kernel(...): ...

def count_local_routed_per_rank(
    topk_ids: Tensor, *, world_size: int, experts_per_rank: int
) -> Tensor: ...
```

### 6.3 `waterfill/__init__.py`

```python
from moe_load_balancer.adapters.sglang.waterfill.runtime import WaterfillRuntime

__all__ = ["WaterfillRuntime"]
```

### 6.4 `waterfill/runtime.py`

```python
class WaterfillRuntime:
    """SGLang adapter runtime for shared-expert Waterfill.

    Replaces PR #19290's DeepEPWaterfillBalancer. Owns:
      - Local routed counts extraction
      - Optional EP all-reduce for dynamic mode
      - Building RoutingRequest and dispatching to WaterfillL2Router
      - Materializing RoutingDecision into SGLang StandardTopKOutput
    """

    MIN_BATCH_FOR_BALANCE = 64

    def __init__(
        self,
        *,
        layer_id: int,
        source_rank: int,
        world_size: int,
        experts_per_rank: int,
        num_routed_experts: int,
        local_preference: float = 1.1,
        routed_scaling_factor: float = 1.0,
        enabled_stages: Optional[frozenset[RoutingStage]] = None,
    ):
        ...
        self.router = WaterfillL2Router(
            source_rank=source_rank,
            local_preference=local_preference,
            world_size=world_size,
        )

    def expand_topk(
        self,
        topk_output,
        num_tokens: int,
        *,
        forward_batch=None,    # ⬅️ pass entire SGLang ForwardBatch; adapter decomposes internally
    ):
        ...
```

The adapter decomposes `forward_batch` internally (extracts `forward_mode`, `num_token_non_padded`, etc. as needed). SGLang glue does not need to know which fields the adapter consumes.

## 7. SGLang Glue Changes

### 7.1 `model_runner.py` (1-line import)

```python
- from moe_load_balancer.adapters.sglang.deepep_waterfill import DeepEPWaterfillBalancer as balancer_cls
+ from moe_load_balancer.adapters.sglang.waterfill import WaterfillRuntime as balancer_cls
```

### 7.2 `topk.py` (forward_batch threading)

`TopK.forward_native` / `forward_cuda` / `forward_cpu` / `forward_npu` / `empty_topk_output` accept an optional `forward_batch` parameter and pass it to `_apply_deepep_waterfill`, which passes it to `balancer.expand_topk(...)`.

### 7.3 `deepseek_v2.py::forward_deepep`

Passes its `forward_batch` argument to `self.topk(...)`.

### 7.4 Other files

`environ.py`, `server_args.py` unchanged.

Total SGLang glue grows from 119 lines (PR #19290) to ~150-180 lines.

## 8. Deletions

| Path | Lines | Reason |
|---|---|---|
| `policies/l2/waterfill.py` `SharedExpertWaterfillPolicy` class | 391 | Replaced by `WaterfillL2Router` (GPU implementation) |
| `adapters/sglang/load.py` (`SGLangRoutingLoadProvider` stub) | 31 | NotImplementedError placeholder; rebuild when LPLB needs it |
| `adapters/sglang/context.py` (`SGLangRoutingContext`) | 28 | Fields already covered by `RoutingRequest`; redundant staging type |
| `adapters/sglang/routing.py` (`SGLangRoutingAdapter`) | 66 | Replaced by `WaterfillRuntime` |
| `adapters/sglang/{placement,waterfill,deepep_waterfill}.py` | 733 | Reorganized into new structure |
| `tests/test_waterfill_policy.py` | 275 | Tests deleted CPU policy class |
| `tests/test_sglang_waterfill_adapter.py` | 78 | Tests deleted `build_*_dispatch_plan` functions |
| Parts of `tests/test_api_contracts.py` | partial | References to deleted classes |

## 9. Documentation Updates

| Document | Change |
|---|---|
| `docs/architecture.md` | `RoutingLoadView/Provider` → renamed; `RoutingLoadHistory` field renames; `RoutingRequest.placement` Optional; new `routed_rank_load` / `active_rank_token_count` fields; `RoutingPolicyConfig.metadata` strictness约定 |
| `docs/sglang-integration.md` | Same renames; `SGLangRoutingContext` section removed (covered by RoutingRequest); `SGLangRoutingAdapter` section refactored to mention `WaterfillRuntime` as canonical PR-aligned adapter |
| `CLAUDE.md` | Already committed: Required Reading section pointing at architecture.md / sglang-integration.md / etc. |

## 10. Pipeline Data Flow (Final)

```
SGLang TopK.forward(... , forward_batch=fb)
  └─► WaterfillRuntime.expand_topk(topk_output, num_tokens, forward_batch=fb):

      [Adapter step] decompose forward_batch -> stage, lb_token_mask, num_token_non_padded

      [Adapter step] count_local_routed_per_rank(topk_ids)  -> local_counts

      [Adapter step] low-batch fast path: if num_tokens < 64,
                     dynamic path still all-reduces for collective consistency,
                     then expand_topk_with_shared_expert (skips fused materialize)

      [Adapter step] mode dispatch:
          static : routed_rank_load = local_counts;            active_rank_token_count = None
          dynamic: routed_rank_load = global_routed (post AR); active_rank_token_count = local_tokens

      [Adapter step] build RoutingRequest with:
          placement=None
          stage, token_count=num_token_non_padded, lb_token_mask, routed_rank_load,
          active_rank_token_count, policy=RoutingPolicyConfig(name="waterfill",
              enabled_stages=self.enabled_stages,
              metadata={"experts_per_rank": ...})

      [Policy step]  decision = WaterfillL2Router.route_tokens(request)
          ├─ stage gating
          ├─ active is None -> static (rank_load=routed, target_total=0 sentinel)
          │  active is not None -> dynamic (rank_load=routed+active, compute target_total)
          ├─ _waterfill_select_kernel
          └─ wrap RoutingDecision

      [Adapter step] materialize_waterfill_dispatch(decision.shared_expert_rank, ...)
                     -> expanded SGLang StandardTopKOutput
```

## 11. Testing

- Delete `tests/test_waterfill_policy.py` and `tests/test_sglang_waterfill_adapter.py`.
- Update `tests/test_api_contracts.py`: remove references to deleted classes; keep `RoutingLoadHistory(Provider)` and `TokenRouter` protocol contract tests using minimal dummy implementers.
- POC stage: GPU correctness validation runs as part of e2e verification (no separate CPU oracle test).
- e2e: SGLang DeepSeek + `--enable-deepep-waterfill`, end-to-end prompt benchmark to confirm parity with pre-restructure throughput.

## 12. Performance Expectation

The refactor preserves PR #19290's performance characteristics:

- **Static path**: PR's `target_total = 0` sentinel + kernel-internal derivation is preserved; no new host-GPU sync.
- **Dynamic path**: PR's host-side `target_total` computation is preserved.
- **Low-batch path**: Static skips all-reduce; dynamic still participates in EP all-reduce for collective consistency, then falls back to local expansion.
- **Kernel split** (count → select → materialize): three Triton launches replace PR's two (count + fused expand). Adds approximately 3-5 μs per layer of launch overhead. For DeepSeek (~60 MoE layers): ~200-300 μs per batch (< 1% prefill, ~3-5% decode).

The asymmetry between static and dynamic `target_total` handling (kernel-derive vs precomputed) is preserved as a deliberate performance optimization, not a refactoring oversight. A symmetric variant was considered (always precompute) but would add a host-GPU sync per layer in static mode (estimated ~9-24% decode latency hit), so it was rejected.

## 13. Migration / Rollout Steps

1. **`moe_load_balancer` repo**:
   1. Update `core/types.py` and `core/l2.py` (new fields, renames).
   2. Update `core/__init__.py` exports.
   3. Rewrite `policies/l2/waterfill.py`.
   4. Create `adapters/sglang/providers.py` and `load_kernels.py`.
   5. Create `adapters/sglang/waterfill/__init__.py` and `runtime.py`.
   6. Delete six old adapter files.
   7. Update `adapters/sglang/__init__.py` for back-compat re-exports.
   8. Update tests (delete two files, modify `test_api_contracts.py`).
   9. Sync `docs/architecture.md` and `docs/sglang-integration.md`.
   10. Run `pytest tests/`; expect green.

2. **SGLang worktree (`sglang-main-moe-lb-waterfill`)**:
   1. Update `model_runner.py` import (1 line).
   2. Extend `topk.py` to accept `forward_batch` parameter.
   3. Extend `deepseek_v2.py::forward_deepep` to pass `forward_batch`.
   4. Amend into `8dd0d498d` glue commit.

3. **Self-test**:
   1. `pip install -e moe_load_balancer --no-deps`.
   2. Verify import paths: `from moe_load_balancer.adapters.sglang.waterfill import WaterfillRuntime`.
   3. Bring up SGLang server with DeepSeek + `--enable-deepep-waterfill`, run a brief sanity prompt.

4. **Push to remote** (after self-test passes).

## 14. Future Work (Out of Scope)

- LPLB integration (`policies/l2/lplb.py` + `adapters/sglang/lplb/runtime.py`) using the same canonical pipeline.
- Real `RoutingLoadHistoryProvider` implementation (when LPLB or historical-stat-based policies need it).
- Stage-aware policy combinations (Waterfill prefill + Metro decode), via SGLang glue selecting different `WaterfillRuntime` / `MetroRuntime` instances based on `server_args`.
- `PolicyName` constant module in `core/types.py` for typo-safety on policy name strings (not required for current POC).

## 15. Open Questions / Risks

- `forward_batch` decomposition relies on stable SGLang `ForwardBatch` field names (`forward_mode`, `num_token_non_padded`, etc.). If SGLang upgrades change these, `WaterfillRuntime.expand_topk` must be updated. This concentrates SGLang-version coupling in one place (acceptable trade-off for the simpler glue).
- `experts_per_rank` flowing through `RoutingPolicyConfig.metadata` is a layout assumption that the policy reads. Other (non-DeepEP) frameworks would set a different value or use a different layout helper; documented in the metadata strict约定.
- The Triton kernel's mode-dependent `target_total` semantics (sentinel vs precomputed) is retained for performance but is non-obvious. Mitigated by docstrings on `WaterfillL2Router.route_tokens` and the kernel itself.
