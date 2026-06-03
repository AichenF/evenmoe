---
title: "Improving DeepEP MoE Load Balance in SGLang with Waterfill and LPLB"
author: "The SGLang Team"
date: "TBD"
previewImg: /images/blog/deepep-waterfill-lplb/preview.jpg
---

## TL;DR

Mixture-of-Experts (MoE) models rely on Expert Parallelism (EP) to scale inference across multiple GPUs. In SGLang, DeepEP provides the high-performance all-to-all dispatch and combine path for EP, but the workload seen by each rank can still be imbalanced because tokens are not routed uniformly across experts.

This blog introduces two dispatch-time load balancing features in SGLang:

- **Waterfill**, a lightweight shared-expert load balancing method for DeepEP. It dispatches the shared expert through DeepEP and assigns it to less-loaded ranks. In our primary two-node H20 DeepSeek-V3/R1-style MMLU serving workload, Waterfill improved total throughput from **29,745 tok/s** to **30,979 tok/s** (**+4.1%**). On DeepSeek V4 Flash, the best measured point improved from **49,253 tok/s** to **51,677 tok/s** (**+4.92%**).
- **LPLB**, a linear-programming-based load balancer for redundant expert replicas. It is designed to solve a per-layer dispatch optimization problem over redundant experts.

The Waterfill work is built on two SGLang PRs: [shared expert fusion](https://github.com/sgl-project/sglang/pull/20089) and [Waterfill dispatch balancing](https://github.com/sgl-project/sglang/pull/19290). DeepSeek V4 Flash support is added in [#25391](https://github.com/sgl-project/sglang/pull/25391). LPLB is introduced in [#24515](https://github.com/sgl-project/sglang/pull/24515).

## Introduction

Large MoE models such as DeepSeek-V3/R1 and DeepSeek V4 Flash use sparse expert activation to increase model capacity while keeping per-token computation manageable. During inference, EP distributes experts across GPUs and routes tokens to the ranks that own the selected experts. This reduces per-GPU memory pressure and makes large-scale serving practical, but it also introduces a central systems problem: **the router does not generate perfectly balanced expert traffic**.

When some experts receive many more tokens than others, the EP group waits for the busiest ranks. This imbalance affects both computation and communication. Static placement methods such as EPLB can improve the long-term placement of experts and redundant replicas, but a single batch can still have residual imbalance. Dispatch-time load balancing addresses this remaining gap by deciding, at runtime, which physical replica should process each token or each shared-expert request.

In SGLang, we have been working on two dispatch-time approaches for DeepEP MoE inference:

- **Waterfill**: a low-overhead algorithm focused on the shared expert path.
- **LPLB**: an LP-based algorithm focused on token routing across redundant expert replicas.

The two algorithms target the same broad layer of the system: dispatch-time MoE load balancing. They make different tradeoffs and operate on different dispatch choices.

## Background: Load Imbalance in DeepEP MoE Inference

DeepEP accelerates MoE inference by providing optimized token dispatch and combine kernels for expert parallelism. In a typical DeepSeek-style MoE layer, each token is routed to several **routed experts** selected by the model router. Some models also include a **shared expert**, which is applied to every token.

From a serving-system perspective, routed experts and shared experts create different load patterns:

- Routed experts are sparse. Different tokens choose different experts, so their load depends on the router distribution.
- Shared experts are dense. Every token needs the shared expert, so the shared-expert workload is present for the full batch.
- Redundant experts, introduced by EPLB-style placement, provide multiple physical replicas for some logical experts. They create an opportunity for dispatch-time balancing, because the system can choose which physical replica processes a token without changing the model's logical expert choice.

Static expert placement is important, but it cannot remove all runtime imbalance. The actual tokens in a batch may still concentrate on a subset of experts or ranks. In DeepEP, this can leave some ranks waiting for overloaded peers. Waterfill and LPLB both aim to reduce this dispatch-time imbalance while preserving the model's semantics.

## Waterfill: Lightweight Load Balancing for Shared Expert Dispatch

### Waterfill Dispatch Strategy

Waterfill is a lightweight load balancing algorithm for the shared expert path under DeepEP.

If the shared expert is always computed locally on every rank, then each rank pays the shared-expert cost regardless of whether it is already overloaded by routed experts. The overloaded ranks remain overloaded, and the less-loaded ranks cannot help absorb the shared-expert work.

Waterfill changes this by treating the shared expert as a dispatchable expert slot. After the routed experts are selected, Waterfill estimates the current routed load on each EP rank, then assigns the shared expert work to ranks with lower load. Conceptually, it fills the valleys in the rank-load distribution, similar to pouring water into uneven containers.

For each token, Waterfill adds one extra expert slot for the shared expert. Instead of always assigning that slot to the token's local rank, it selects a rank based on the current load distribution. This keeps the routed expert choices unchanged, so the model still computes the same logical routed experts and the same shared expert. The only thing that changes is **which physical rank executes the shared expert work**.

There is an important communication tradeoff. If every token could send its
shared-expert work to any EP rank, Waterfill would have more balancing freedom,
but it could also increase all-to-all traffic. For GPU MoE serving, communication
is often more expensive than the extra shared-expert computation. Waterfill
therefore keeps the candidate ranks local to the token's existing dispatch
footprint: in the communication-sensitive path, the shared expert is assigned
only to ranks that the token already visits for routed experts, with the source
rank kept as a fallback. This preserves most of the load-balancing benefit while
keeping communication overhead close to the original DeepEP dispatch path.

By shifting less shared-expert work to already-heavy ranks and more shared-expert work to lighter ranks, Waterfill makes the per-rank workload more balanced and improves end-to-end system throughput.

### Shared Expert Fusion as the Enabling Mechanism

Waterfill depends on shared expert fusion.

Before shared expert fusion, SGLang computed the shared expert separately from the DeepEP MoE dispatch path. The routed experts went through DeepEP dispatch, grouped GEMM, and combine, while the shared expert was launched as a separate local computation on every rank. That design is straightforward, but it has two limitations for Waterfill:

1. The shared expert is not dispatchable. If it does not go through DeepEP, Waterfill has no dispatch decision to make.
2. The system pays an additional execution path for the shared expert. The shared expert is launched separately from the routed expert path, adding overhead and making it harder to balance the work with routed experts.

Shared expert fusion fixes this by representing the shared expert as an extra expert slot in the MoE path. In DeepSeek V3/R1, this is commonly described as adding a "9th expert" slot per rank: the original top-k routed experts remain unchanged, and one additional column is appended for the shared expert. Once the shared expert is part of the DeepEP dispatch/compute/combine pipeline, Waterfill can choose where that shared-expert slot is executed.

This is why the Waterfill feature was split into two pieces:

- [#20089](https://github.com/sgl-project/sglang/pull/20089) fuses the shared expert into the DeepEP MoE path with a fixed local assignment.
- [#19290](https://github.com/sgl-project/sglang/pull/19290) adds Waterfill, which replaces the fixed assignment with load-aware shared-expert dispatch.

The fusion itself is not the final load balancing algorithm. It is the required mechanism that makes shared-expert dispatch visible to DeepEP and therefore controllable by Waterfill.

## LPLB: LP-Based Load Balancing for Redundant Expert Replicas

### LPLB Routing Strategy

TODO: This section will be completed by the LPLB owners.

Suggested points to cover:

- What optimization problem LPLB solves.
- How LPLB uses redundant expert replicas.
- How it differs from Waterfill.
- Why it is useful for per-layer dispatch-time balancing.

## Evaluation

### Waterfill on DeepSeek V3/R1

We evaluated Waterfill on a DeepSeek-V3/R1-style setup with FP8 weights, two H20 nodes, 16 GPUs total, EP16, DP16, DeepEP normal mode, and DP attention. The serving workload uses real MMLU prompts generated from the SGLang MMLU benchmark. For the original PR reproduction, the MMLU prompt set was split into two non-overlapping files: 2,000 prompts for recording the EPLB distribution and 2,000 prompts for serving throughput.

The primary serving benchmark uses:

- Dataset: MMLU prompts, 5-shot, all subjects generated through `benchmark/mmlu/bench_sglang.py`.
- Prompt length: approximately 600 input tokens on average; measured rounds usually contain about 650K-675K total tokens for 1,000 requests.
- Request shape: `max_tokens=1`, `temperature=0`.
- Batch shape: 1,000 sampled prompts per measured round.
- Concurrency: 256 concurrent requests.
- Metric: total token throughput, counting prompt tokens plus the single generated token.
- Statistics: warmup rounds are discarded; measured rounds report a trimmed mean after dropping the highest and lowest throughput samples.

The table below summarizes the DeepSeek-V3/R1-style Waterfill results. The
non-redundant red0 row is from the original PR reproduction using the 2,000
prompt serving split. The red16 and red32 rows are newer redundant-EPLB measurements
on the full 14,042-prompt MMLU pool. All rows use 1,000 requests per measured
round, concurrency 256, and `max_tokens=1`.

| Configuration | Physical Routed Experts | Baseline | Waterfill | Gain |
| --- | ---: | ---: | ---: | ---: |
| Static EPLB, red0 | 256 | 29,745 tok/s | 30,979 tok/s | +4.1% |
| Static EPLB, red16 | 272 | 31,070 tok/s | 31,472 tok/s | +1.29% |
| Static EPLB, red32 | 288 | 30,942 tok/s | 32,153 tok/s | +3.91% |

For reference, standalone DeepEP shared-expert fusion without Waterfill measured 30,569 tok/s for red0, 30,992 tok/s for red16, and 31,124 tok/s for red32 under the same 14,042-prompt, batch=1,000, concurrency=256, `max_tokens=1` workload. This separates the shared-expert fusion baseline from the Waterfill dispatch policy.

These results indicate that Waterfill improves throughput while preserving model quality, because it only changes physical shared-expert placement and does not change the logical expert computation.

### Waterfill on DeepSeek V4 Flash

DeepSeek V4 Flash uses a `HashTopK` routing path, so Waterfill support also needs to
cover that path. [#25391](https://github.com/sgl-project/sglang/pull/25391)
extends Waterfill to DeepSeek V4 Flash `HashTopK`.

Two-node H20 DeepSeek V4 Flash FP8 showed consistent throughput improvement on the same MMLU-style serving workload shape: 14,042-prompt pool, batch=512, concurrency=128, `max_tokens=1`, 2 warmup rounds, and 4 measured rounds. The table reports trimmed mean total throughput.

| Configuration | Baseline | Waterfill | Gain |
| --- | ---: | ---: | ---: |
| No EPLB | 45,951 tok/s | 47,876 tok/s | +4.19% |
| Static EPLB, red0 | 49,253 tok/s | 51,677 tok/s | +4.92% |
| Static EPLB, red16 | 50,006 tok/s | 51,655 tok/s | +3.30% |
| Static EPLB, red32 | 50,167 tok/s | 51,813 tok/s | +3.28% |

These results use the V4 Flash `HashTopK` path and validate that Waterfill remains
directionally positive on DeepSeek V4 Flash.

### LPLB on DeepSeek R1

TODO: This section will be completed by the LPLB owners.

Suggested points to cover:

- Benchmark setup.
- Dynamic baseline versus LP dispatch.
- Throughput and TTFT results.
- Workload conditions where the measured signal is strongest.

### Accuracy Validation

Waterfill preserves model semantics because it does not alter the router's logical top-k decisions. The routed experts selected by the model remain the same, and the shared expert remains the same shared expert. Waterfill only changes which physical EP rank executes the shared expert slot.

For LPLB, the expected invariant is similar: dispatch-time routing should select among valid redundant physical replicas for the same logical expert. The LPLB owners can fill in the exact validation methodology and results.

## How to Use

### Enable Waterfill

Waterfill is enabled through the DeepEP MoE path. A representative launch command is:

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/DeepSeek-V3 \
    --tp 16 \
    --dp-size 16 \
    --nnodes 2 \
    --node-rank ${NODE_RANK} \
    --dist-init-addr ${HEAD_NODE_IP}:${PORT} \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --enable-dp-attention \
    --enable-deepep-waterfill \
    --init-expert-location /path/to/expert_distribution.pt
```

The important flags are:

- `--moe-a2a-backend deepep`: use DeepEP for MoE all-to-all dispatch.
- `--enable-deepep-waterfill`: enable the shared expert fusion and Waterfill path.
- `--init-expert-location`: optionally initialize expert placement and rank-load metadata from collected expert distribution statistics.

DeepSeek V4 Flash support uses the `HashTopK` path added in
[#25391](https://github.com/sgl-project/sglang/pull/25391).

### Enable LPLB

TODO: This section will be completed by the LPLB owners.

Suggested command shape:

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/DeepSeek-R1 \
    --trust-remote-code \
    --tp-size 16 \
    --dp-size 16 \
    --ep-size 16 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --ep-num-redundant-experts 16 \
    --ep-dispatch-algorithm lp \
    --init-expert-location /path/to/expert_stats.pt
```

## Acknowledgment

This work builds on the SGLang DeepEP and MoE serving stack and was developed through community collaboration in the SGLang project.

We thank the SGLang maintainers and reviewers for discussions, reviews, and integration support across the related PRs:

- [#20089: Fuse shared expert into MoE dispatch under EP](https://github.com/sgl-project/sglang/pull/20089)
- [#19290: Add Waterfill load balancing for shared expert dispatch](https://github.com/sgl-project/sglang/pull/19290)
- [#25391: Support DeepSeek V4 DeepEP Waterfill](https://github.com/sgl-project/sglang/pull/25391)
- [#24515: LPLB: linear-programming load balancer for MoE expert parallelism](https://github.com/sgl-project/sglang/pull/24515)
