# LPLB / LP-CF User Guide

LP-CF is the current L2 physical-routing policy used for the LPLB stage. It
maps logical TopK expert IDs to physical replicas using the committed
placement and the current global logical-expert load.

LP-CF is a framework-neutral policy. Framework lifecycle, collectives and
output materialization are not part of the policy.

## Requirements

- a committed `PlacementSnapshot`;
- logical TopK IDs and weights;
- per-logical-expert counts prepared by the framework;
- CUDA and Triton for the production router implementation.

For a combined LPLB + Waterfill path, the framework performs the LPLB
logical-count all-reduce before calling MLB. The orchestrator computes the
post-LPLB per-rank physical load locally and then runs Waterfill.

## Construct the unified balancer

```python
from moe_load_balancer import MoELoadBalancer
from moe_load_balancer.policies.l2.lpcf import LPCFL2Router
from moe_load_balancer.policies.l2.waterfill import WaterfillL2Router

mlb = MoELoadBalancer(
    routing_policies={
        "lplb": LPCFL2Router(
            ep_size=ep_size,
            num_physical_experts=num_physical_experts,
        ),
        "waterfill": WaterfillL2Router(
            source_rank=ep_rank,
            world_size=ep_size,
        ),
    },
)
```

The policy objects belong to the one generic `MoELoadBalancer` instance.
There is no `LPCFRuntime` or `WaterfillRuntime`.

## Build an SGLang request

SGLang Glue first prepares the collective result and obtains a snapshot from
the live committed metadata:

```python
from moe_load_balancer import RoutingPolicyConfig
from moe_load_balancer.adapters.sglang import (
    to_placement_snapshot,
    to_routing_request,
)

snapshot = to_placement_snapshot(expert_location_metadata, layer_id)

request = to_routing_request(
    layer_id=layer_id,
    logical_topk_ids=topk_ids,
    topk_weights=topk_weights,
    placement=snapshot,
    global_logical_count=global_logical_count,
    policies=(
        RoutingPolicyConfig(name="lplb"),
        RoutingPolicyConfig(
            name="waterfill",
            metadata={"experts_per_rank": num_local_physical_experts},
        ),
    ),
    stage="decode",
)
```

`global_logical_count` must already have the semantics required by LP-CF.
The adapter does not count tokens and does not run all-reduce.

## Route once

```python
decision = mlb.route_tokens(request)
```

The internal order is:

```text
LP-CF
→ physical IDs
→ per-rank physical load count
→ Waterfill
→ RoutingDecision
```

The combined path deliberately clears `active_rank_token_count` before
Waterfill so it uses the post-LPLB static load and does not trigger a second
communication.

## Write back to SGLang

```python
from moe_load_balancer.adapters.sglang import to_sglang_routing_output

output = to_sglang_routing_output(
    decision,
    router_logits=router_logits,
    num_routed_experts=num_routed_experts,
    world_size=ep_size,
    routed_scaling_factor=routed_scaling_factor,
)
```

`output.recorded_physical_topk_ids` is the routed-expert recorder input.
`output.topk_ids` and `output.topk_weights` contain the SGLang-facing output;
when Waterfill ran, the shared expert has been appended and the IDs use the
DeepEP interleaved layout.

The SGLang Glue is still responsible for constructing its
`StandardTopKOutput` and invoking dispatch.

## Empty ranks

The MLB call itself performs no collective. SGLang must still execute its
logical-count collective on every rank, including ranks with zero local
tokens, before constructing the request.

## Tests

Run the full CPU suite:

```bash
pytest -q
```

The NumPy LP-CF oracle runs without Triton. CUDA/Triton kernel parity tests
are skipped when CUDA is unavailable.

## Integration status

This repository now provides the target MLB API and stateless SGLang
translations. It does not modify the SGLang fork or consolidate the SGLang
TopK call sites; that framework wiring is a separate change.
