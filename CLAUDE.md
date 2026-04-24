# MoE Load Balancer - Agent Notes

面向用户的入口是 `README.md`。本文件只给 coding agent 快速建立上下文。项目材料整理时，优先更新 `README.md`，不要把主线重新收窄成单一 L2 demo。

## Source of Truth

- Google Doc: https://docs.google.com/document/d/1xeDxh2sH1atr6DBnoO6QkXjZ8EKTcWa4emsq4Hk76Dw/edit?tab=t.0#heading=h.d3248k5k32jx

## Agent Workflow

- 每次 coding agent 完成文件修改后，都必须检查 diff/status，并创建 git commit。

## Correct Project Shape

目标是构建框架解耦的 MoE Load Balancer，先在 SGLang 中开发验证，再集成到 TensorRT-LLM MoE plugin，使 NeMo Inference 和 Dynamo 间接受益。

三层能力:

- **L1 Static Expert Placement**: 基于历史 routing stats 决定 expert 放置和每层副本数。
- **L2 Dynamic Token Routing**: 每 batch/每层决定 token 发给同一逻辑 expert 的哪个物理副本。
- **L3 Prediction + Expert Prefetch**: 预测下一层 expert 激活模式，提前做 P2P expert 权重复制。

关键约束:

- 精度无损: 不改变 router top-k 逻辑 expert，只改变同一逻辑 expert 的物理副本选择。
- 阶段感知: prefill 均衡 token 数，decode 均衡激活 expert 数。
- 框架解耦: 核心模块不依赖 SGLang/vLLM/TensorRT-LLM 内部类型。
- 能力可独立开关: L1/L2/L3 都应能单独启用或组合使用。

## Roadmap

| Stage | 内容 | 交付物 |
|---|---|---|
| Stage 0 | 已有基础 | EPLB in SGLang、Waterfill PR、Shared Expert Fusion、LPLB 进行中 |
| Stage 1 | 调研 + 工程并行 | 三层技术调研、LPLB benchmark、SGLang 修改点、vLLM/TRT-LLM hook 点 |
| Stage 2 | 架构设计 | 统一模块边界、接口层次、框架 adapter 方案 |
| Stage 3 | SGLang POC | L1 静态放置原型 + L2 prefill Waterfill/LPLB；明确不做 L3 和 decode/Metro |
| Stage 4 | 模块化 + 框架集成 | 独立模块、TensorRT-LLM 集成、vLLM 适配 |

如果有短期 SGLang demo 计划，应把它写成 Stage 3 的一个最小切片，而不是替代 proposal 的完整 roadmap。当前 POC 的 L2 只做 prefill Waterfill + LPLB；L3 prediction/prefetch 和 decode Metro 都不进入 POC。

## Canonical Interfaces

proposal 级接口:

```python
class MoELoadBalancer:
    def compute_placement(self, routing_stats, num_gpus, hbm_budget) -> PlacementPlan:
        ...

    def route_tokens(self, gate_output, placement, stage: "prefill" | "decode") -> TokenMapping:
        ...

    def predict_and_prefetch(self, hidden_states, next_layer_gate) -> PrefetchPlan:
        ...
```

SGLang 里的 `modify_topk()` 可以是 `route_tokens()` 的 adapter/helper，但不应作为项目级主接口写进顶层设计。

## Existing SGLang Work

| PR | 标题 | 状态 |
|---|---|---|
| [#20089](https://github.com/sgl-project/sglang/pull/20089) | Shared Expert Fusion | 已合入，commit `57ffc55fb` |
| [#22525](https://github.com/sgl-project/sglang/pull/22525) | EPLB Compatible | 待补充 |
| [#19290](https://github.com/sgl-project/sglang/pull/19290) | Waterfill Load Balancing | Under review，端到端吞吐约 +4% |

团队并行工作:

- LPLB 集成到 SGLang: DeepSeek Layer-wise Per-batch Load Balancing。
- LPLB pipeline: Count -> Solve -> Map，LP 求解约 100us/batch。

## Stage 1 Notes

调研应覆盖三层:

- L1: EPLB、CRAFT、MoETuner，重点是 per-layer replica budget 和通信感知 placement。
- L2: Waterfill、LPLB、Metro、HarMoEny，重点是 prefill/decode 目标差异和 Waterfill+LPLB 可组合性。当前 POC 实现只取 prefill Waterfill + LPLB，Metro 仅保留为后续调研项。
- L3: Libra、PROBE，重点是预测准确率、P2P 传输隐藏条件和跨节点限制。当前 POC 不实现 L3。

必须补齐代码级调研:

- Waterfill / LPLB PR 与 SGLang 集成修改点。
- vLLM / TensorRT-LLM 中 EPLB 或 MoE plugin 的模块落点、调用链、hook 点。

## Known Corrections

- 早期调研笔记曾写 CRAFT 已在 SGLang v0.4.8 实现。
- 核查 SGLang main 后，上游没有 CRAFT 实现；`eplb_algorithms/` 只有 `deepseek`、`deepseek_hierarchical`、`deepseek_vec`、`deepseek_vec_hierarchical`、`elasticity_aware`。
- 原论文提到的 SGLang v0.4.8 实现应为作者 fork，未合入上游。
