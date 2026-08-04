# MoE Load Balancer

`moe-load-balancer` is an alpha-stage Python library for framework-decoupled
Mixture-of-Experts load balancing. It provides one entry point for two levels
of decisions:

- **L1 placement planning** chooses which logical expert weights should occupy
  each physical expert slot.
- **L2 token routing** chooses among equivalent physical replicas and, when
  configured, selects a destination rank for a shared expert.

Frameworks retain ownership of model execution, collectives, recording,
rebalance cadence, weight movement, and committed placement metadata. Adapters
translate framework-native values to and from the plain request and result
types used by this package.

## Status

The current release is `0.1.0` and the public API is still evolving.

| Level | Policy | Status | Purpose |
|---|---|---|---|
| L1 | `auto` | Implemented | Selects the appropriate implemented DeepSeek EPLB mode from topology. |
| L1 | `deepseek` | Implemented | Non-hierarchical expert placement. |
| L1 | `deepseek_hierarchical` | Implemented | Node-aware hierarchical expert placement. |
| L2 | `lplb` | Implemented | LP-based routing among redundant physical replicas. |
| L2 | `waterfill` | Implemented | Shared-expert destination selection from per-rank routed load. |

`deepseek_vec` and `elasticity_aware` source placeholders are retained for
future work, but they are intentionally not registered as supported policies.

## Architecture

```text
Serving framework Glue
    │
    ├── framework Adapter: native metadata/tensors → MLB requests
    │
    ▼
MoELoadBalancer
    ├── L1 placement policy → uncommitted PlacementPlan
    └── ordered L2 policy pipeline → RoutingDecision
            ├── LPLB physical-replica routing
            └── Waterfill shared-expert routing
    │
    ├── framework Adapter: MLB result → native values
    │
    ▼
Serving framework applies the result
```

The main source directories are:

| Path | Responsibility |
|---|---|
| `src/moe_load_balancer/core/` | Public request/result types, protocols, and the unified orchestrator. |
| `src/moe_load_balancer/policies/l1/` | Framework-independent placement algorithms. |
| `src/moe_load_balancer/policies/l2/` | Framework-independent routing policies. |
| `src/moe_load_balancer/kernels/` | Triton and CUDA execution implementations. |
| `src/moe_load_balancer/adapters/sglang/` | Stateless translation between SGLang and MLB values. |
| `examples/` | Executable examples used by CI. |
| `tests/` | API, policy, adapter, orchestrator, and GPU integration tests. |

## Installation

Python 3.10 or newer, PyTorch 2.5 or newer, Triton, tvm-ffi, and NVIDIA
MathDX are required. The normal installation installs these dependencies once;
the Waterfill and LPLB kernels themselves are included in the MLB package.

Install the complete package:

```bash
python -m pip install -e .
```

The LPLB CUDA implementation requires a compatible NVIDIA CUDA toolchain.
`nvidia-mathdx` is pinned to the version used by the packaged JIT sources. A
manual MathDX installation can instead be selected with `MATHDX_HOME`.

Inside an existing SGLang environment, dependency versions are normally owned
by that environment:

```bash
python -m pip install -e /path/to/moe_load_balancer --no-deps
```

## Quick Start: L1 Placement

The following is also available as `examples/quickstart.py` and is executed by
CI:

```python
import torch

from moe_load_balancer import MoELoadBalancer
from moe_load_balancer.adapters.sglang import to_placement_request

request = to_placement_request(
    torch.tensor([[20, 3, 9, 4], [5, 16, 2, 7]]),
    num_physical_experts=8,
    num_local_physical_experts=4,
    num_groups=1,
    num_nodes=1,
    algorithm="auto",
)

plan = MoELoadBalancer().plan_placement(request)

print(plan.physical_to_logical_map.shape)
print(plan.metadata["resolved_policy"])
```

`PlacementPlan` is only a candidate. The serving framework must move or load
weights first and update its live metadata before exposing the resulting
placement to L2 as a committed `PlacementSnapshot`.

## Configuring the L2 Pipeline

The framework constructs the runtime-specific policy instances and registers
them on one `MoELoadBalancer` instance:

```python
from moe_load_balancer import MoELoadBalancer
from moe_load_balancer.kernels.lplb import CUDALPLBKernels
from moe_load_balancer.policies.l2.lplb import LPLBL2Router
from moe_load_balancer.policies.l2.waterfill import WaterfillL2Router

world_size = 16
source_rank = 0

mlb = MoELoadBalancer(
    routing_policies={
        "lplb": LPLBL2Router(
            kernels=CUDALPLBKernels(),
            num_gpus=world_size,
        ),
        "waterfill": WaterfillL2Router(
            source_rank=source_rank,
            world_size=world_size,
        ),
    }
)
```

For a combined request, the ordered policy sequence is `("lplb",
"waterfill")`. The orchestrator runs physical-replica routing first, counts the
resulting per-rank load, and then runs shared-expert Waterfill. A framework may
also register and run either policy independently.

SGLang Glue is responsible for:

1. Reading TopK output and live placement metadata.
2. Completing any required EP all-reduce before entering MLB.
3. Calling the stateless SGLang adapter functions to create a
   `PlacementRequest`, `PlacementSnapshot`, or `RoutingRequest`.
4. Calling `MoELoadBalancer.plan_placement()` or
   `MoELoadBalancer.route_tokens()`.
5. Translating the result and applying it inside SGLang.

## Development

Install development dependencies and run the local quality gates:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy src/moe_load_balancer
python -m pytest -m unit
python examples/quickstart.py
python -m build
```

GPU integration tests are separate because they require a CUDA device and the
matching CUDA toolchain:

```bash
python -m pytest -m integration
```

## Compatibility and Performance

The current CUDA integration has been smoke-tested on NVIDIA H20 GPUs in the
corresponding SGLang integration environment. GPU compatibility depends on the
PyTorch, CUDA, Triton, tvm-ffi, and MathDX versions used by the serving
framework; new combinations should run the integration test suite before
deployment.

## License

MoE Load Balancer is licensed under Apache-2.0. Copied and adapted DeepSeek and
SGLang components retain their original notices and licenses; see
`THIRD_PARTY_NOTICES.md`.
