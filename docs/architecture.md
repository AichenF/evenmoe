# MoE Load Balancer Architecture

本文档把 proposal 中的三层架构整理成工程边界。它不是单独的 L2 PoC 说明，而是 Stage 2 架构设计的基础稿。

## Problem

大规模 MoE 推理通常使用 Expert Parallelism。router 的 token-to-expert 分配天然不均，热点 expert 会让部分 GPU 过载，而其他 GPU 空闲等待，端到端吞吐受最慢 GPU 限制。

已有方案分散在不同层:

| 层面 | 代表方案 | 主要问题 |
|---|---|---|
| 静态放置 | EPLB、CRAFT、MoETuner | 副本预算和 placement 分散实现，容易浪费 HBM |
| 动态路由 | Waterfill、LPLB、Metro | prefill/decode 目标不同，尚无统一接口 |
| 预测预取 | Libra、PROBE | 执行流更复杂，对 GPU 互联带宽敏感 |

目标是把这些能力收敛到一个框架解耦模块中，通过 adapter 接入 SGLang、TensorRT-LLM、vLLM 等推理框架。

## Design Goals

- **Framework independent**: 核心模块不 import 推理框架内部类型。
- **Orthogonal layers**: L1/L2/L3 可独立开关，也可组合使用。
- **Lossless correctness**: 不改变 router top-k 逻辑 expert；routed expert 重映射只能在同一逻辑 expert 的物理副本内选择，shared expert Waterfill 只能追加等价的 shared dispatch slot。
- **Stage aware but not stage-bound**: prefill 和 decode 的优化目标可能不同，但不是所有算法都必须按 stage 分叉。`RoutingRequest.stage` 是 optional runtime context；`RoutingPolicyConfig.enabled_stages` 是 core 级通用 gating，允许一个 policy 只在 prefill、只在 decode、两者都运行，或完全不按 stage 限制。`enabled_stages=None` 不要求 stage 已知；显式 stage set 要求 `request.stage` 已知且匹配。当前 SGLang POC 只复用 SGLang EPLB 作为 L1 placement source；新的 L1 optimizer、Metro 留作后续。
- **Product path**: SGLang first validation，TensorRT-LLM MoE plugin productization。

## Core Vs Adapter Boundary

核心模块的输入输出必须是框架中立的逻辑数据:

- L1 当前只从 SGLang EPLB 导入 `PlacementSnapshot`。新的 `PlacementPlan` / planner 不进入当前 POC。
- L2 输入 logical top-k ids、top-k weights、placement snapshot 和 optional stage，输出 replica/physical expert 选择。
- L3 输入框架中立的 hidden-state/gate view，输出 prefetch plan。

框架 adapter 负责把这些 plan 落到具体系统对象上。例如 SGLang adapter 才能接触 `ExpertLocationMetadata`、`ExpertLocationDispatchInfo`、`StandardTopKOutput`、DeepEP process group、interleaved expert id 和 shared-expert fusion layout。核心算法不能 import 这些类型，也不能直接发起 SGLang 通信 API。

## Architecture Overview

```text
MoE Load Balancer
├── Layer 3: Prediction + Expert Prefetch
│   ├── input: current hidden states, next layer gate
│   └── output: next-layer replica/prefetch plan
├── Layer 2: Dynamic Token Routing
│   ├── input: logical top-k, placement, optional stage
│   └── output: token -> selected replica / physical expert mapping
├── Layer 1: SGLang EPLB Placement Source
│   ├── input: SGLang ExpertLocationMetadata
│   └── output: framework-neutral PlacementSnapshot
└── Framework Adapters
    ├── SGLang
    ├── TensorRT-LLM
    └── vLLM
```

The framework adapter owns framework-specific objects and call sites. The core load balancer owns algorithms and common data contracts.

## Layer 1: Static Expert Placement

执行频率: 离线、模型加载时或低频重算。

当前职责: 复用 SGLang 已有 EPLB placement metadata、init/rebalance 和 weight update 路径，作为 L2 routing 的 placement source 和 benchmark baseline。

当前 POC 不决定新的 expert placement，也不实现每层副本预算。

后续如果重新打开 L1 optimizer，再单独讨论以下子问题:

- **Replica budget allocation**: 不同层负载偏斜度不同，应给高偏斜层更多副本，低偏斜层少给或不给，避免均匀副本策略浪费 HBM 并挤压 KV cache。
- **Placement optimization**: 在副本数确定后，把高激活 expert 分散到不同 GPU，并考虑通信量，把高通信 expert pair 尽量放在高带宽链路附近。

候选基础:

- EPLB: 已有 SGLang 基础。
- CRAFT / MoETuner: 作为 per-layer replica budget 和 placement 优化参考。

这些不属于当前 POC。当前 L2 只消费从 EPLB metadata 转换来的 `PlacementSnapshot`。

## Layer 2: Dynamic Token Routing

执行频率: 每 batch 或每 MoE 层，在 MoE dispatch 之前。

职责: 在保持逻辑 top-k 不变的前提下改写 dispatch routing。对 routed experts，只能在同一逻辑 expert 的物理副本内选择；对 PR #19290 对齐的 Waterfill，只追加 shared expert dispatch slot。

阶段感知:

- **Prefill**: compute-bound，常见目标是均衡各 GPU token 数。
- **Decode**: memory-bound，常见目标是均衡各 GPU 激活 expert 数。

SGLang 中的 Waterfill/LPLB 工作属于 L2 的已有基础和 Stage 1 工程并行项。Waterfill 可以作为 stage-agnostic L2 policy 单独使用；是否只在 prefill、只在 decode 或两者运行由 policy config 决定。当前 Stage 3 POC 不做 Metro。

## Layer 3: Prediction + Expert Prefetch

执行频率: 每层，可选。

职责: 利用当前层 hidden states 预测下一层 expert 激活模式，并提前通过 P2P 复制 expert 权重。

当前 SGLang POC 不实现 L3。L3 只保留在长期架构和 Stage 1/2 调研设计中。

核心机制:

- 投机执行下一层 gating，利用相邻层 hidden states 缓慢变化的性质。
- 生成下一层 expert 复制方案。
- 用双缓冲把 P2P 传输隐藏在计算中。

适用条件:

- 更适合 NVSwitch 等高带宽 GPU 互联。
- 跨节点 InfiniBand 场景下权重传输可能无法被计算掩盖，因此 L3 必须可关闭。

## Canonical API

proposal 级接口保持三层语义:

```python
@dataclass
class RoutingRequest:
    layer_id: int
    logical_topk_ids: Tensor
    topk_weights: Tensor
    placement: PlacementSnapshot
    stage: Optional[Literal["prefill", "decode", "mixed", "idle", "speculative"]]
    token_count: Optional[Tensor]
    lb_token_mask: Optional[Tensor]
    routed_scaling_factor: float
    policy: RoutingPolicyConfig


@dataclass
class RoutingPolicyConfig:
    name: str
    # None means the policy is stage-agnostic and may run with or without stage.
    enabled_stages: Optional[set[Literal["prefill", "decode", "mixed", "idle", "speculative"]]]


@dataclass
class RoutingDecision:
    routed_physical_topk_ids: Tensor
    topk_weights: Tensor
    shared_expert_rank: Optional[Tensor] = None
    stats: Optional[RoutingStats] = None


class MoELoadBalancer:
    def route_tokens(
        self,
        request: RoutingRequest,
    ) -> RoutingDecision:
        ...

    def predict_and_prefetch(
        self,
        request: PrefetchRequest,
    ) -> PrefetchPlan:
        ...
```

`logical_topk_ids` 是 token-level top-k，不是 logical expert 聚合分布。L2 必须结合 `PlacementSnapshot` 中的 physical/rank mapping 计算 routed rank load；shared-expert Waterfill 不能只看 logical expert count。

SGLang adapter 另提供 `PlacementProvider.snapshot(layer_id) -> PlacementSnapshot`。SGLang 可以有 `modify_topk()` / `route_topk()` 这样的局部 helper，但它应被视为 `route_tokens()` 在 SGLang TopK 数据结构上的 adapter，不是架构级接口。

## Module Boundary

建议的长期模块结构:

```text
moe_load_balancer/
├── api.py
├── config.py
├── types.py
├── placement/
│   └── snapshot.py
├── routing/
│   ├── waterfill.py
│   ├── lplb.py
│   └── metro.py
├── prefetch/
│   ├── predictor.py
│   └── planner.py
└── adapters/
    ├── sglang/
    │   ├── placement.py
    │   └── routing.py
    ├── tensorrt_llm/
    └── vllm/
```

Stage 3 POC 可以先把核心代码 vendored 到 SGLang 内部目录，但核心逻辑仍应按上述边界组织，避免后续抽独立 package 时重写。
当前 Stage 3 的 L1 目录/接口只需要支持 EPLB placement snapshot adapter；replica budget 和 placement optimizer 不进入当前 POC。

## Integration Path

```text
Stage 1: research + current SGLang engineering
Stage 2: define common interfaces and adapter split
Stage 3: validate SGLang EPLB-backed L2 prototype
Stage 4: extract independent module and integrate TensorRT-LLM
```

SGLang 是验证平台；TensorRT-LLM 是产品化目标；NeMo Inference 和 Dynamo 通过后端引擎集成间接受益。
