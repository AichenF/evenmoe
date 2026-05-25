# LP-CF User Guide

LP-CF (closed-form pair-split) is an L2 token-routing algorithm for MoE
models with redundant expert replicas. For each redundant logical expert it
splits the per-batch token mass across physical replicas in closed form
— no LP / IPM solve. Two Triton kernels, no CUDA C++, no external solver.

This guide covers how to run it under SGLang and how to test it.

---

## When to use LP-CF

- The model has redundant experts (`--ep-num-redundant-experts > 0`).
- You are using EP / DP-attention (per-rank LB hot path matters).
- You want a low-overhead solver replacement for the IPM-based LPLB.

LP-CF is a sibling of LPLB (not a kernel inside it). LPLB is preserved on
a separate branch and not part of `main`.

---

## Requirements

| | |
|---|---|
| moe_load_balancer | `main` branch, commit `5c20ca0` or later |
| sglang fork | `feliang-git/sglang`, branch `mlb-glue-lpcf` |
| Hardware | NVIDIA GPU with Triton support (validated on H20) |
| Python | `triton`, `torch` (whatever the sglang dev container has) |

---

## Install

Inside the SGLang dev container:

```bash
# 1. moe_load_balancer (this repo) — editable install
git clone https://github.com/xutizhou/moe_load_balancer.git
cd moe_load_balancer
git checkout main
pip install -e .

# 2. SGLang fork with the LP-CF glue
git clone https://github.com/feliang-git/sglang.git
cd sglang
git checkout mlb-glue-lpcf
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

No CUDA build step is needed — both Triton kernels JIT-compile on first use.

---

## Run

Add `--ep-dispatch-algorithm lpcf` to the SGLang launch line. Minimum
example (DeepSeek-R1, 2-node EP=16):

```bash
python -m sglang.launch_server \
    --model-path /raid/model/DeepSeek-R1 --trust-remote-code \
    --tp-size 16 --dp-size 16 --ep-size 16 \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --dist-init-addr <node0_ip>:29501 \
    --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode normal \
    --ep-num-redundant-experts 16 \
    --ep-dispatch-algorithm lpcf \
    --init-expert-location /path/to/expert_stats.pt \
    ...
```

### Mode switch: local vs. global counts

| Mode | Env var | What it does |
|---|---|---|
| **Local count** (default) | `LPCF_USE_GLOBAL_COUNT=0` (or unset) | Each rank's solver runs on its own DP shard counts. Zero collective overhead per layer. Recommended default. |
| Global count | `LPCF_USE_GLOBAL_COUNT=1` | Per-layer EP all-reduce of logical-expert counts before the solve. Slightly better load balance in some workloads; pays the per-layer collective cost. |

Set the env var before launching SGLang. Every rank must agree on the
value — under `=1`, every rank must call into the runtime on every layer
(including empty-token ranks under DP-attention) so the collective
completes. The SGLang glue already wires the empty-rank participation.

### Empty-rank discipline

Under DP-attention some ranks see zero tokens for a given MoE layer. The
glue (`models/deepseek_v2.py`) drives `LPCFRuntime.route` on an empty
`StandardTopKOutput` from those branches, so:

- With `LPCF_USE_GLOBAL_COUNT=0`: cheap no-op (count_local_logical short-circuits).
- With `LPCF_USE_GLOBAL_COUNT=1`: empty rank still participates in the all-reduce.

This is handled automatically — nothing to configure.

---

## Test

### Unit tests (CPU-only oracle + GPU Triton parity)

```bash
cd moe_load_balancer
pytest tests/test_lpcf.py -v
```

10 tests:
- 7 numpy oracle property tests (CPU; run in any env).
- 1 Triton CF kernel parity test (skipped without CUDA).
- 2 Triton dispatch_probability sanity tests (skipped without CUDA).

All 10 pass on H20 in the SGLang dev container.

### Per-layer smoke (no SGLang server)

Spin up an `LPCFL2Router` and drive a synthetic `RoutingRequest`:

```python
import torch
from moe_load_balancer.policies.l2.lpcf import LPCFL2Router
from moe_load_balancer.core.types import (
    PlacementSnapshot, RoutingRequest, RoutingPolicyConfig,
)

EP_SIZE = 16
NUM_PHYS = 272   # 256 logical + 16 redundant on R1
device = "cuda"

router = LPCFL2Router(ep_size=EP_SIZE, num_physical_experts=NUM_PHYS)

# Fake placement: every logical has 1 replica on a round-robin physical.
log2phy = torch.full((256, NUM_PHYS), -1, dtype=torch.int32, device=device)
log2phy[:, 0] = torch.arange(256, dtype=torch.int32, device=device)
log_counts = torch.ones(256, dtype=torch.int32, device=device)
placement = PlacementSnapshot(
    layer_id=0,
    num_logical_experts=256,
    num_physical_experts=NUM_PHYS,
    logical_to_physical_candidates=log2phy,
    logical_to_physical_count=log_counts,
)

topk_ids = torch.randint(0, 256, (1024, 8), dtype=torch.int32, device=device)
counts = torch.zeros(256, dtype=torch.int32, device=device)
counts.scatter_add_(0, topk_ids.flatten().long(), torch.ones_like(topk_ids).flatten())

decision = router.route_tokens(RoutingRequest(
    layer_id=0,
    logical_topk_ids=topk_ids,
    topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
    placement=placement,
    global_logical_count=counts,
    policy=RoutingPolicyConfig(name="lpcf"),
))
print("physical:", decision.routed_physical_topk_ids.shape, decision.routed_physical_topk_ids.dtype)
```

The router will JIT-compile the two Triton kernels on first call (one-shot
cost, then cached for the process lifetime).

### End-to-end SGLang smoke

Launch the server with `--ep-dispatch-algorithm lpcf`, hit `/health` until
200, send a completion request. A reference script (parameterized for an
H20 dev cluster) lives at `scripts/lpcf_sglang_smoke.sh` and exits 0 on
success. Expected runtime once JIT cache is warm: ~80 s for server up +
one 16-token completion.

### Benchmark

The `scripts/cf_lc_vs_dyn_3rep_ds.sh` script in this repo runs a 3-rep
counterbalanced A/B between LP-CF (`LPCF_USE_GLOBAL_COUNT=0`) and the
SGLang stock `dynamic` algorithm on a single dataset. Validated config:
gsm8k / mmlu / sharegpt at `--max-concurrency 4096`.

Reference results (DeepSeek-R1, 2-node EP=16, 3-rep avg):

| Dataset | dyn input tok/s | LP-CF input tok/s | Δ | dyn Mean TTFT | LP-CF Mean TTFT | Δ |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k    | 18,440 | **19,514** | **+5.8%** | 8,400  | **8,267**  | −1.6% |
| mmlu     | 18,952 | **21,201** | **+12%**  | 14,565 | **13,405** | −8.0% |
| sharegpt |  1,960 |  **1,989** | +1.5%     | 50,196 | **46,837** | −6.7% |

Report **input throughput** and **TTFT** — those are the metrics LP-CF
moves. TPOT / output-tok/s are decode-stage metrics and not the right
yardstick for an L2 prefill optimizer.

---

## Where the code lives

| Concern | File |
|---|---|
| Triton CF kernel + numpy oracle | `src/moe_load_balancer/policies/l2/lpcf.py` |
| Triton dispatch_probability kernel | same file (`_dispatch_probability_kernel`) |
| Triton fused `_count_local_logical` | `src/moe_load_balancer/adapters/sglang/lpcf/runtime.py` |
| `LPCFL2Router` (L2 algorithm) | `src/moe_load_balancer/policies/l2/lpcf.py` |
| `LPCFRuntime` (SGLang adapter) | `src/moe_load_balancer/adapters/sglang/lpcf/runtime.py` |
| Tests | `tests/test_lpcf.py` |

SGLang glue (separate repo `feliang-git/sglang`, branch `mlb-glue-lpcf`):

| Concern | File |
|---|---|
| Per-layer runtime registry | `python/sglang/srt/eplb/moelb_lpcf_registry.py` |
| Dispatch-info wiring | `python/sglang/srt/eplb/expert_location_dispatch.py` |
| Per-layer call site | `python/sglang/srt/layers/moe/topk.py` |
| Startup hook | `python/sglang/srt/model_executor/model_runner.py::_init_lpcf_runtimes` |
| Empty-rank discipline | `python/sglang/srt/models/deepseek_v2.py` |
| CLI flag + preflight | `python/sglang/srt/server_args.py` |

---

## Algorithm sketch

For each logical expert `i` with `K_i` redundant physical replicas:

- `K_i = 1`: all tokens for logical `i` go to its single replica.
- `K_i ≥ 3`: uniform `G_i / K_i` per replica. (R1 placements typically
  have <5% logicals at K≥3, so this is a small fraction.)
- `K_i = 2`: split `p · G_i` to replica A, `(1−p) · G_i` to B with
  ```
  p = clip( (L_b − L_a + G_i) / (2 · G_i), 0, 1 )
  ```
  where `L_{a,b}` is the post-(K=1, K≥3) rank load snapshot. Same-rank
  pairs degenerate to `p = 0.5`.

All K=2 logicals read the same load snapshot (parallel sweep), then a
multinomial-sample kernel maps each `(token, slot)` from logical to
physical id. Both phases are one Triton launch each, plus one launch
for the per-batch logical-count.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `RuntimeError: --ep-dispatch-algorithm=lpcf requires the moe_load_balancer package` | The SDK isn't on the `PYTHONPATH` / not pip-installed. |
| `topk_ids_logical_to_physical called with ep_dispatch_algorithm='lpcf'` | The per-layer hook in `topk.py::_post_process_topk_ids` isn't reaching the LP-CF branch — check the call ordering. |
| EP all-reduce deadlock under DP-attention | You set `LPCF_USE_GLOBAL_COUNT=1` on an older SGLang fork without the empty-rank participation hook. Use `mlb-glue-lpcf` branch. |
| First-call latency spike | Triton JIT compile (~hundreds of ms for the first call; cached afterward). Run a warmup request before benchmarking. |
