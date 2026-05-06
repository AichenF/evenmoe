# MoE Load Balancer

本目录是 Google Doc proposal 的本地整理和后续工程工作台。

Source of truth:

- Google Doc: https://docs.google.com/document/d/1xeDxh2sH1atr6DBnoO6QkXjZ8EKTcWa4emsq4Hk76Dw/edit?tab=t.0#heading=h.d3248k5k32jx

## Project Intent

构建一个框架解耦的 MoE 推理负载均衡器，用系统化方式解决 expert/GPU 负载不均衡问题。模块先在 SGLang 中开发和验证，再抽象为独立组件，并集成到 TensorRT-LLM MoE plugin，使 NeMo Inference 和 Dynamo 间接受益。

这个项目不是单一算法集成，也不只是 Waterfill + LPLB demo。proposal 的主线是三层正交能力:

1. **Layer 1: SGLang EPLB Placement Source**
   当前 POC 只复用 SGLang 现有 EPLB placement metadata / rebalance 路径，不新增 placement optimizer。
2. **Layer 2: Dynamic Token Routing**
   每 batch/每层生成 dispatch routing；routed expert 只能在同一逻辑 expert 的物理副本内选择，PR 对齐的 Waterfill 只追加 shared expert dispatch slot。
3. **Layer 3: Prediction + Expert Prefetch**
   预测下一层 expert 激活模式，并提前做 P2P expert 权重复制。

所有优化都必须保持精度无损: 不改变 router top-k 逻辑 expert，只改变同一逻辑 expert 的物理副本选择。

## Roadmap

| Stage | 目标 | 交付物 |
|---|---|---|
| Stage 0 | 梳理已有基础 | EPLB in SGLang、shared-expert Waterfill PR、Shared Expert Fusion、LPLB 进行中 |
| Stage 1 | 调研 + 工程并行 | 三层技术调研报告、LPLB benchmark、Waterfill/LPLB SGLang 修改点、vLLM/TRT-LLM hook 点 |
| Stage 2 | 架构设计 | 统一模块边界、接口规范、框架 adapter 方案 |
| Stage 3 | SGLang POC | 复用 SGLang EPLB 作为 placement source + L2 shared-expert Waterfill/LPLB；具体实验可通过 policy config 限制在 prefill、decode 或两者；不包含新 L1 optimizer、L3 和 Metro |
| Stage 4 | 模块化 + 框架集成 | 独立 load balancer 模块、TensorRT-LLM 集成和 benchmark |

当前文档整理应优先服务 Stage 1 和 Stage 2: 先把三层问题、现有实现、hook 点和统一接口讲清楚，再进入 Stage 3 的 SGLang POC。当前 POC 范围已经收紧: L1 只复用 SGLang EPLB，不做新的 replica budget / placement optimizer；L2 聚焦 shared-expert Waterfill + LPLB，是否只在 prefill、只在 decode 或两者都启用由 core 级 policy config 决定；L3 prediction/prefetch 和 Metro 先不做。

## Canonical Interfaces

当前 POC 的标准接口是:

```python
class PlacementProvider:
    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        ...


class MoELoadBalancer:
    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        ...

    def predict_and_prefetch(self, request: PrefetchRequest) -> PrefetchPlan:
        ...
```

`PlacementProvider` 在 SGLang POC 中由 EPLB adapter 实现；`RoutingRequest` 使用框架中立的数据，例如 logical top-k ids、top-k weights、placement snapshot 和 optional stage；`RoutingPolicyConfig.enabled_stages` 是 core 级通用 stage gating，`None` 表示该算法不按 stage 限制，也允许 `stage=None`。显式列出所有 stage 则仍要求调用方提供已知 stage。`RoutingDecision` 返回框架中立的 routing decision，再由 adapter 写成 physical/interleaved expert id 或 shared expert slot。SGLang 里的 `StandardTopKOutput`、`ExpertLocationMetadata`、process group、DeepEP interleaved id 等只能出现在 adapter/helper 中，不能成为项目级主接口。

## Repository Skeleton

当前 repo 已提供一个最小 L1/L2 skeleton，方便并行推进 API 设计和 SGLang 接入点调研:

- `src/moe_load_balancer/core/`: 框架中立 API contract。
- `src/moe_load_balancer/policies/l1/`: L1 placement policy 实现目录。
- `src/moe_load_balancer/policies/l2/`: L2 routing policy 实现目录。
- `src/moe_load_balancer/adapters/sglang/`: SGLang adapter 占位，负责 SGLang 类型转换、TopK writeback 和 runtime glue。

## Working Assumptions

- SGLang 是首个开发与验证平台，不是最终唯一目标。
- TensorRT-LLM MoE plugin 是最重要的产品化集成目标。
- NeMo Inference 和 Dynamo 通过后端引擎集成间接受益。
- 三层能力应可独立开关，也可组合使用。
- Prefill 和 decode 的优化目标不同: prefill 侧重均衡 token 数，decode 侧重均衡激活 expert 数；但这不是所有算法的强制分类轴。某个 L2 policy 可以通过 `enabled_stages` 限制只在 prefill、只在 decode、两者都运行，或完全不按 stage gating。
