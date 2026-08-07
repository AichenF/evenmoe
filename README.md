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

## End-to-End Serving Performance

The table below shows observed end-to-end SGLang serving throughput with a
DeepSeek-V3 FP8 10-layer integration checkpoint. Each value is total throughput
in K tokens/s; the parenthesized value is the observed difference from the
Baseline in the same dataset. **Bold** marks the highest observed throughput in
each row. `All` enables EPLB, Waterfill, and LPLB together.

| Dataset | Baseline | Waterfill | LPLB | EPLB | EPLB + Waterfill | EPLB + LPLB | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K-1K | 161.77 | 165.48 (+2.29%) | 159.72 (-1.27%) | 188.69 (+16.64%) | 190.03 (+17.47%) | 189.44 (+17.11%) | **191.44 (+18.34%)** |
| MMLU-1K | 183.96 | 189.04 (+2.76%) | 180.82 (-1.71%) | 193.47 (+5.17%) | **194.27 (+5.61%)** | 192.62 (+4.71%) | 192.86 (+4.84%) |
| GPQA-1K | 187.91 | 192.53 (+2.46%) | 182.51 (-2.87%) | 193.53 (+2.99%) | **195.00 (+3.77%)** | 192.90 (+2.66%) | 192.71 (+2.55%) |
| ShareGPT-1K | 192.71 | **194.88 (+1.12%)** | 189.69 (-1.57%) | 193.48 (+0.40%) | 194.42 (+0.89%) | 192.10 (-0.32%) | 192.37 (-0.18%) |

In this experiment, standalone Waterfill had a positive observed difference on
all four datasets. EPLB produced the largest differences on GSM8K, while
EPLB + Waterfill had the highest observed throughput on MMLU and GPQA. The
highest observed ShareGPT result used Waterfill alone. Standalone LPLB was
below the Baseline in these four workloads; its purpose is replica routing and
its benefit depends on the placement and request distribution.

Benchmark protocol:

- MLB commit `80c44ed83055ffa9332e8d7d254fa3abfa4c0f62` with SGLang commit
  `a380a107ed705b18e1f512b03f55b3c2970e61b4`.
- Two exclusive nodes with 8 NVIDIA H20 96 GB GPUs per node,
  TP16/DP16/EP16; GPU clocks were not locked.
- DeepSeek-V3 FP8 checkpoint with a 10-layer configuration overlay. These are
  integration-benchmark results, not full-model production-serving numbers.
- Fixed 1,024-token inputs, a disjoint 2,000-prompt benchmark set replayed in
  the same order, concurrency 256, `max_tokens=1`, and `temperature=0`.
- Three fresh-server groups per dataset and mode. Each group used four nominal
  warmup rounds, an explicit convergence gate, and eight measured rounds. The
  reported value is the median of the three group medians.
- 672 measured rounds and 1,344,000/1,344,000 completed measured requests,
  with zero request errors and zero invalid output-token counts.
- Immutable, verified per-dataset EPLB artifacts; CUDA Graph enabled, DeepEP
  auto mode, and the overlap scheduler disabled.
- All modes passed the predefined 3% gates. The worst within-group CV was
  2.57%, and the worst range across the three fresh-server medians was 0.72%.

The relative values above are descriptive differences for this pinned
end-to-end serving setup. Bold identifies the largest observed value, not
statistical significance between close alternatives. These results are not a
hardware-normalized kernel SOL measurement.

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

## Reproducible EPLB Experiments

An EPLB performance comparison must reuse one immutable recording artifact.
Re-recording the same prompts can change small token counts and therefore
change the discrete expert placement, which creates a different experimental
cohort rather than a repeat of the original run.

After recording once, seal the recording together with the record/benchmark
datasets and request schedule:

```bash
python scripts/eplb_repro_artifact.py seal \
  --manifest artifacts/eplb_manifest.json \
  --record-dataset artifacts/record_2k.jsonl \
  --benchmark-dataset artifacts/benchmark_2k.jsonl \
  --schedule artifacts/benchmark_schedule.json \
  --recording artifacts/logical_count.pt \
  --num-physical-experts 288 \
  --num-local-physical-experts 18 \
  --num-groups 8 --num-nodes 2 --algorithm auto \
  --identity model=DeepSeek-V3 --identity profile=gsm8k_1k
```

Every server launch in the comparison must pass verification before using the
recording with SGLang's `--init-expert-location`:

```bash
python scripts/eplb_repro_artifact.py verify artifacts/eplb_manifest.json
```

The manifest binds the raw file bytes, canonical `logical_count` tensor,
topology, resolved policy, and all derived placement maps. It is immutable:
sealing an existing path with different inputs fails instead of silently
overwriting the reference placement. Use `compare` to confirm that two staged
experiment directories belong to the same EPLB cohort.

For short-input serving benchmarks, also replay every held-out request in the
same order and at the same concurrency on every round. Use fresh server
launches, alternate the A/B launch order, and compare within-launch-group
medians. The manifest prevents placement drift; the fixed replay protocol
prevents asynchronous request completion from silently changing the continuous
batch sequence. Record the client and launch configuration alongside the
manifest.

## Compatibility

The current CUDA integration has been smoke-tested on NVIDIA H20 GPUs in the
corresponding SGLang integration environment. GPU compatibility depends on the
PyTorch, CUDA, Triton, tvm-ffi, and MathDX versions used by the serving
framework; new combinations should run the integration test suite before
deployment.

## License

MoE Load Balancer is licensed under Apache-2.0. Copied and adapted DeepSeek and
SGLang components retain their original notices and licenses; see
`THIRD_PARTY_NOTICES.md`.
