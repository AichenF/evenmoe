# MoE Load Balancer Architecture

## Goal

This repository separates framework lifecycle from framework-neutral MoE
load-balancing policies.

The scaling target is:

```text
M framework-neutral policies + N framework adapters
```

It must not become M×N framework-and-policy runtimes.

## Ownership

Frameworks such as SGLang own:

- model and engine lifecycle;
- TopK/dispatcher recording hooks;
- forward boundaries, windows, cadence and threshold decisions;
- distributed collectives;
- authoritative placement metadata;
- weight loading, migration, recovery and commit;
- backend-specific output materialization.

MLB owns:

- framework-neutral requests, decisions and plans;
- L1 placement policies;
- ordered L2 routing policies;
- a single `MoELoadBalancer` entry point;
- stateless translations between framework data and MLB data.

MLB does not store framework runtime objects, recorder windows or a second
copy of committed placement state.

## Dependency direction

The framework chooses and invokes its adapter:

```text
Framework Glue
    ↓
pre Adapter
    ↓
MoELoadBalancer
    ↓
post Adapter
    ↓
Framework Glue
```

`MoELoadBalancer` never accepts `framework="sglang"` and never imports a
framework adapter.

## Unified entry point

One model or engine uses one generic instance:

```python
from moe_load_balancer import MoELoadBalancer

mlb = MoELoadBalancer(
    placement_planners={...},
    routing_policies={...},
)

candidate = mlb.plan_placement(placement_request)
decision = mlb.route_tokens(routing_request)
```

Policy implementations may cache framework-neutral GPU buffers, but there is
no per-framework/per-algorithm runtime or process-global registry.

## L1: candidate placement

### Input

The framework finishes recording, aggregation, all-reduce and rebalance
gating before MLB is called. A stateless adapter then builds:

```python
PlacementRequest(
    stats=PlacementStats(logical_count=...),
    policy=PlacementPolicyConfig(
        name="eplb",
        metadata={
            "algorithm": ...,
            "num_physical_experts": ...,
            "num_local_physical_experts": ...,
            "num_groups": ...,
            "num_nodes": ...,
        },
    ),
)
```

### Output

`PlacementPlan` contains only candidate mapping tensors:

```python
PlacementPlan(
    physical_to_logical_map=...,
    logical_to_all_physical_map=...,
    logical_to_physical_count=...,
)
```

It does not contain `PlacementSnapshot` and is not visible to L2.

### Commit rule

```text
PlacementRequest
→ MLB L1 Policy
→ candidate PlacementPlan
→ framework moves/loads real weights
→ framework commits placement metadata
→ later L2 calls observe the new PlacementSnapshot
```

If weight movement fails, the framework keeps its old committed placement and
L2 continues using that placement.

Cold start and periodic rebalance use the same MLB planning entry point. They
differ only in the framework trigger.

## PlacementSnapshot

`PlacementSnapshot` is a read-only view of the framework's currently
committed per-layer placement:

- logical and physical expert counts;
- EP size and local physical expert count;
- physical-to-logical mapping;
- logical-to-physical candidates and replica counts;
- optional default logical-to-physical mapping;
- optional explicit physical-to-rank mapping.

The SGLang adapter is a stateless function:

```python
snapshot = to_placement_snapshot(live_metadata, layer_id)
```

It stores no metadata and performs no cache invalidation. Repeating the call
after SGLang commits new metadata returns a view of the new layout.

## L2: ordered policy pipeline

`RoutingRequest.policies` is an ordered tuple. Policies declare a phase:

- `RoutingPhase.PHYSICAL_ROUTING`: logical expert IDs to physical expert IDs;
- `RoutingPhase.SHARED_EXPERT`: shared-expert placement after physical routing.

The supported target order is:

```text
logical TopK
→ optional LPLB/LP-CF physical routing
→ count post-LPLB physical load per rank
→ optional Waterfill shared-expert selection
→ RoutingDecision
```

Only one physical-routing policy and one shared-expert policy run in one
pipeline. A physical-routing policy cannot run after a shared-expert policy.

### Combined LPLB + Waterfill

In the combined path:

1. the framework supplies the all-reduced logical counts required by LPLB;
2. LPLB returns physical expert IDs;
3. the orchestrator counts those physical IDs per rank locally;
4. Waterfill consumes that new rank load;
5. no second collective is performed for Waterfill.

### Waterfill without LPLB

The framework may provide already-routed physical IDs and dynamic rank-load
tensors. Otherwise, the orchestrator obtains physical IDs from the committed
placement's default mapping and counts them locally.

### Recorder boundary

`RoutingDecision.routed_physical_topk_ids` is the recorder-facing result.
Shared-expert materialization happens later, so the shared expert is not
included in EPLB routed-expert statistics.

## SGLang adapters

The adapter package exposes stateless functions:

```text
adapters/sglang/placement.py
    to_placement_request
    to_sglang_maps
    to_placement_snapshot

adapters/sglang/routing.py
    to_routing_request
    to_sglang_routing_output
```

They do not:

- dump or reset a recorder;
- perform all-reduce;
- decide cadence or threshold;
- call a weight updater;
- retain placement metadata;
- own `MoELoadBalancer`;
- import or select policies.

## Source layout

```text
src/moe_load_balancer/
├── core/
│   ├── types.py
│   ├── l1.py
│   ├── l2.py
│   └── orchestrator.py
├── policies/
│   ├── l1/eplb/
│   └── l2/
│       ├── lpcf.py
│       └── waterfill.py
└── adapters/
    └── sglang/
        ├── placement.py
        └── routing.py
```

SGLang-side call-site consolidation and the future SGLang-owned aggregate
state object are outside this repository change.
