---
title: "Improving DeepEP MoE Load Balance in SGLang with Waterfill and LPLB"
author: "The SGLang Team"
date: "TBD"
previewImg: /images/blog/deepep-waterfill-lplb/preview.jpg
---

## TL;DR

Mixture-of-Experts (MoE) models rely on Expert Parallelism (EP) to scale inference across multiple GPUs. In SGLang, DeepEP provides the high-performance all-to-all dispatch and combine path for EP, but the workload seen by each rank can still be imbalanced because tokens are not routed uniformly across experts.

This blog introduces two dispatch-time load balancing features in SGLang:

- **Waterfill**, a lightweight shared-expert load balancing method for DeepEP. It dispatches the shared expert through DeepEP and assigns it to less-loaded ranks, improving throughput by about **4%** on DeepSeek-V3/R1-style workloads and up to **4.31%** in our DeepSeek V4 sanity tests.
- **LPLB**, a linear-programming-based load balancer for redundant expert replicas. It is designed to solve a per-layer dispatch optimization problem over redundant experts.

The Waterfill work is built on two SGLang PRs: [shared expert fusion](https://github.com/sgl-project/sglang/pull/20089) and [Waterfill dispatch balancing](https://github.com/sgl-project/sglang/pull/19290). DeepSeek V4 support is added in [#25391](https://github.com/sgl-project/sglang/pull/25391). LPLB is introduced in [#24515](https://github.com/sgl-project/sglang/pull/24515).

## Introduction

Large MoE models such as DeepSeek-V3/R1 and DeepSeek V4 use sparse expert activation to increase model capacity while keeping per-token computation manageable. During inference, EP distributes experts across GPUs and routes tokens to the ranks that own the selected experts. This reduces per-GPU memory pressure and makes large-scale serving practical, but it also introduces a central systems problem: **the router does not generate perfectly balanced expert traffic**.

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

### What Waterfill Is and Why It Works

Waterfill is a lightweight load balancing algorithm for the shared expert path under DeepEP.

The key observation is simple: when routed expert load is imbalanced across ranks, the shared expert does not have to make the imbalance worse. If the shared expert is always computed locally on every rank, then each rank pays the shared-expert cost regardless of whether it is already overloaded by routed experts. The overloaded ranks remain overloaded, and the less-loaded ranks cannot help absorb the shared-expert work.

Waterfill changes this by treating the shared expert as a dispatchable expert slot. After the routed experts are selected, Waterfill estimates the current routed load on each EP rank, then assigns the shared expert work to ranks with lower load. Conceptually, it fills the valleys in the rank-load distribution, similar to pouring water into uneven containers.

For each token, Waterfill adds one extra expert slot for the shared expert. Instead of always assigning that slot to the token's local rank, it selects a rank based on the current load distribution. This keeps the routed expert choices unchanged, so the model still computes the same logical routed experts and the same shared expert. The only thing that changes is **which physical rank executes the shared expert work**.

The result is a small but consistent improvement in end-to-end throughput for workloads where shared-expert execution amplifies EP imbalance.

### Why Waterfill Requires Shared Expert Fusion

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

### What LPLB Is and How It Works

TODO: This section will be completed by the LPLB owners.

Suggested points to cover:

- What optimization problem LPLB solves.
- How LPLB uses redundant expert replicas.
- How it differs from Waterfill.
- Why it is useful for per-layer dispatch-time balancing.

## Evaluation

### Waterfill on DeepSeek V3/R1

We evaluated Waterfill on a DeepSeek-V3-style setup with FP8 weights, 2 nodes, 16 H20 GPUs, EP16, and DP attention. The benchmark uses MMLU-style prompts with approximately 600 input tokens, one output token, and 256 concurrent requests across multiple rounds.

| Configuration | Output Throughput | Gain |
| --- | ---: | ---: |
| Baseline + static EPLB | ~29,700 tok/s | - |
| Waterfill + static EPLB | ~30,900 tok/s | +4% |

The accuracy sanity check showed no degradation:

| Configuration | MMLU Accuracy |
| --- | ---: |
| Baseline | 91.55% |
| Waterfill + static EPLB | 91.80% |

These results indicate that Waterfill improves throughput while preserving model quality, because it only changes physical shared-expert placement and does not change the logical expert computation.

### Waterfill on DeepSeek V4

DeepSeek V4 introduces additional engineering requirements for the Waterfill path, including `HashTopK`, logical/physical expert IDs, and redundant physical experts. [#25391](https://github.com/sgl-project/sglang/pull/25391) extends Waterfill support to this path.

Two-node H20 DeepSeek V4 FP8 sanity tests showed consistent throughput improvement:

| Configuration | Baseline | Waterfill | Gain |
| --- | ---: | ---: | ---: |
| no-red, batch=512, concurrency=128 | 46,028 tok/s | 47,224 tok/s | +2.60% |
| red16, batch=512, concurrency=128 | 48,666 tok/s | 50,049 tok/s | +2.84% |
| red32, batch=512, concurrency=128 | 48,538 tok/s | 50,630 tok/s | +4.31% |

For an MMLU sanity run with 1000 examples:

| Configuration | Score | Output Throughput |
| --- | ---: | ---: |
| Baseline | 0.878 | 892.9 tok/s |
| Waterfill | 0.876 | 902.6 tok/s |

The small score difference is within the expected range for this sanity test, while the throughput remains directionally positive.

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

For DeepSeek V4, Waterfill additionally needs the V4 TopK/HashTopK and logical/physical expert ID path introduced in [#25391](https://github.com/sgl-project/sglang/pull/25391).

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
