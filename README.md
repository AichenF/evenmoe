# MoE Load Balancer

`moe_load_balancer` is a prototype Python package for framework-decoupled
Mixture-of-Experts (MoE) load balancing. It separates algorithm policy code
from framework glue so the same placement/routing ideas can be validated in
SGLang first and later reused by other runtimes such as TensorRT-LLM, vLLM,
NeMo Inference, or Dynamo.

The repo currently focuses on two layers:

- **L1 placement planning**: framework-neutral expert placement plans. The
  current implementation wraps the DeepSeek/SGLang EPLB placement algorithm.
- **L2 token routing**: per-layer, per-batch routing decisions that preserve
  logical router output while choosing physical expert replicas or shared
  expert dispatch targets. Current policies include shared-expert Waterfill
  and LP-CF.

The core rule is accuracy preservation: policies must not change the router's
logical top-k expert choices. They may only choose among equivalent physical
replicas or add framework-supported shared-expert dispatch metadata.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/moe_load_balancer/core/` | Framework-neutral dataclasses and protocols. |
| `src/moe_load_balancer/policies/l1/` | L1 placement planning policies, currently EPLB. |
| `src/moe_load_balancer/policies/l2/` | L2 routing policies, currently Waterfill and LP-CF. |
| `src/moe_load_balancer/adapters/sglang/` | SGLang-specific conversion, runtime glue, and Triton helper kernels. |
| `tests/` | Unit tests for core API contracts and LP-CF behavior. |
| `docs/architecture.md` | Design notes and module boundaries. |
| `docs/lpcf-user-guide.md` | LP-CF usage guide. |
| `docs/h20-pr19290-reproduction-howto.md` | Waterfill reproduction notes for the H20/SGLang PR workflow. |
| `docs/deepep-waterfill-lplb-blog-draft.md` | Blog draft for Waterfill and LPLB work. |

## Install

For local development:

```bash
git clone https://github.com/xutizhou/moe_load_balancer.git
cd moe_load_balancer
pip install -e ".[dev]"
```

For use inside an SGLang worktree or dev container:

```bash
pip install -e /path/to/moe_load_balancer --no-deps
```

The core package has no required runtime dependencies. GPU policy/runtime
modules import `torch` and `triton` only when those modules are imported.

## Run Tests

```bash
pytest
```

Useful narrower runs:

```bash
pytest tests/test_api_contracts.py -v
pytest tests/test_lpcf.py -v
```

Some LP-CF tests are CUDA/Triton parity checks and are skipped automatically
when CUDA is unavailable.

## Use The Core API

Framework-independent callers should depend on `core` types and policy
protocols, not on SGLang classes:

```python
from moe_load_balancer.core.types import RoutingPolicyConfig, RoutingRequest, RoutingStage

request = RoutingRequest(
    layer_id=0,
    stage=RoutingStage.PREFILL,
    logical_topk_ids=topk_ids,
    topk_weights=topk_weights,
    placement=placement_snapshot,
    policy=RoutingPolicyConfig(name="lpcf"),
)

decision = router.route_tokens(request)
```

The adapter is responsible for translating framework-native tensors and
metadata into these neutral request objects, and for writing the returned
decision back into the framework's native dispatch format.

## Use With SGLang

This repo is meant to be imported as a package by an SGLang integration
branch instead of being copied into the SGLang source tree.

In an SGLang dev environment:

```bash
pip install -e /path/to/moe_load_balancer --no-deps
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH
```

SGLang-side glue should stay small and framework-specific:

- read SGLang metadata such as `ExpertLocationMetadata`;
- build `PlacementSnapshot` / `RoutingRequest`;
- call the selected policy/runtime in this package;
- write the `RoutingDecision` back to SGLang top-k, physical expert ids, or
  shared-expert dispatch slots.

Algorithm code that does not depend on SGLang internals should live in
`src/moe_load_balancer/policies/`. SGLang-only conversion code should live in
`src/moe_load_balancer/adapters/sglang/` or in the SGLang integration branch.

## Current Algorithms

### EPLB Placement

`EPLBPlacementPlanner` exposes the DeepSeek/SGLang EPLB algorithm family
through the framework-neutral `PlacementPlanner` protocol. L1 policy names
directly select `deepseek`, `deepseek_vec`, `elasticity_aware`, their
hierarchical variants, or `auto`; the selected algorithm is not hidden in
metadata. The planner consumes logical expert activation statistics and
returns an uncommitted `PlacementPlan`.

### Waterfill Routing

`WaterfillL2Router` implements shared-expert Waterfill routing. It does not
change routed top-k experts. It consumes per-rank routed load and optional
active token counts, then chooses the shared-expert destination rank.

Waterfill is intentionally usable with or without stage gating. A framework
adapter may run it only in prefill, only in decode, in both, or without
passing stage information.

### LP-CF Routing

`LPCFL2Router` is a closed-form L2 routing policy for logical experts with
redundant physical replicas. See `docs/lpcf-user-guide.md` for the SGLang
launch flow, environment variables, tests, and benchmark guidance.

## Development Notes

- Keep framework-neutral API contracts in `core`.
- Keep algorithm logic in `policies`.
- Keep SGLang, DeepEP, process-group, interleaved-id, and native writeback
  logic in adapters or SGLang glue.
- Do not put per-step tensors in policy `metadata`; pass them through
  `RoutingRequest` or a provider interface.
- Stage is optional context. Policies should support ungated execution unless
  a specific algorithm truly requires stage-specific behavior.
