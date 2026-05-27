# MoE Load Balancer

# English

This repository is the local engineering workspace for the Google Doc proposal and the follow-up implementation work.

Source of truth:

- Google Doc: https://docs.google.com/document/d/1xeDxh2sH1atr6DBnoO6QkXjZ8EKTcWa4emsq4Hk76Dw/edit?tab=t.0#heading=h.d3248k5k32jx

## Project Intent

Build a framework-decoupled MoE inference load balancer that systematically addresses expert/GPU load imbalance. The module is first developed and validated in SGLang, then abstracted into an independent component and integrated into the TensorRT-LLM MoE plugin, so NeMo Inference and Dynamo can benefit through their backend engines.

This project is not a single-algorithm integration and is not just a Waterfill + LPLB demo. The proposal is organized around three orthogonal layers:

1. **Layer 1: SGLang EPLB Placement Source**
   The current POC only reuses SGLang's existing EPLB placement metadata / rebalance path. It does not introduce a new placement optimizer.
2. **Layer 2: Dynamic Token Routing**
   Generate dispatch routing for each batch and each layer. Routed experts can only be selected among physical replicas of the same logical expert. The PR-aligned Waterfill path only appends a shared-expert dispatch slot.
3. **Layer 3: Prediction + Expert Prefetch**
   Predict the next layer's expert activation pattern and proactively perform P2P expert-weight replication.

All optimizations must preserve lossless accuracy: they must not change the router's logical top-k experts, and may only change physical replica selection for the same logical expert.

## Roadmap

| Stage | Goal | Deliverables |
|---|---|---|
| Stage 0 | Baseline study | EPLB in SGLang, shared-expert Waterfill PR, Shared Expert Fusion, LPLB in progress |
| Stage 1 | Research + engineering in parallel | Three-layer technical research report, LPLB benchmark, Waterfill/LPLB SGLang integration points, vLLM/TRT-LLM hook points |
| Stage 2 | Architecture design | Unified module boundaries, interface specification, framework adapter plan |
| Stage 3 | SGLang POC | Reuse SGLang EPLB as the placement source + L2 shared-expert Waterfill/LPLB; experiments can be limited by policy config to prefill, decode, or both; does not include a new L1 optimizer, L3, or Metro |
| Stage 4 | Modularization + framework integration | Independent load balancer module, TensorRT-LLM integration, and benchmark |

The current documentation work should primarily serve Stage 1 and Stage 2: first clarify the three-layer problem, existing implementations, hook points, and unified interfaces, then move into the Stage 3 SGLang POC. The current POC scope has been narrowed: L1 only reuses SGLang EPLB and does not build a new replica budget / placement optimizer; L2 focuses on shared-expert Waterfill + LPLB, with core-level policy config deciding whether a policy runs only in prefill, only in decode, in both, or without stage gating; L3 prediction/prefetch and Metro are deferred.

## Canonical Interfaces

The current POC uses the following canonical interfaces:

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

In the SGLang POC, `PlacementProvider` is implemented by the EPLB adapter. `RoutingRequest` uses framework-neutral data such as logical top-k ids, top-k weights, placement snapshot, and optional stage. `RoutingPolicyConfig.enabled_stages` is a core-level generic stage gate. `None` means the algorithm is not restricted by stage and also allows `stage=None`. Explicitly listing all stages still requires the caller to provide a known stage. `RoutingDecision` returns a framework-neutral routing decision, which the adapter then writes back as physical/interleaved expert ids or shared-expert slots. SGLang-specific types such as `StandardTopKOutput`, `ExpertLocationMetadata`, process groups, and DeepEP interleaved ids may only appear in adapters/helpers, and must not become project-level primary interfaces.

## Repository Skeleton

This repo currently provides a minimal L1/L2 skeleton to support parallel API design and SGLang integration-point research:

- `src/moe_load_balancer/core/`: framework-neutral API contracts.
- `src/moe_load_balancer/policies/l1/`: L1 placement policy implementations.
- `src/moe_load_balancer/policies/l2/`: L2 routing policy implementations.
- `src/moe_load_balancer/adapters/sglang/`: SGLang adapter placeholder, responsible for SGLang type conversion, TopK writeback, and runtime glue.

## Working Assumptions

- SGLang is the first development and validation platform, not the final or only target.
- The TensorRT-LLM MoE plugin is the most important production integration target.
- NeMo Inference and Dynamo benefit indirectly through backend-engine integration.
- The three layers should be independently switchable and also composable.
- Prefill and decode have different optimization goals: prefill focuses on balancing token count, while decode focuses on balancing activated expert count. However, this is not a mandatory classification axis for every algorithm. An L2 policy can use `enabled_stages` to run only in prefill, only in decode, in both, or without stage gating.

## SGLang PR Safety

SGLang PR #19290 is an independent PR and must remain independently reviewable, testable, and mergeable. Follow-up integration experiments for this module must not directly modify or push to the real PR #19290 branch.

Development rules:

- The PR #19290 branch should only contain the Waterfill implementation required by that PR. It must not import this not-yet-ready `moe_load_balancer` module.
- Modular integration experiments must use a separate worktree / branch, for example `pr-19290-moe-lb-integration`, and must not overwrite `feat/deepep-waterfill-eplb-balance` or the PR branch.
- During remote validation, the SGLang integration worktree should only apply the experimental patch. The module code should be installed as an independent package, for example `pip install -e /path/to/moe_load_balancer --no-deps`.
- Unless we explicitly decide to change the PR strategy, do not push the import-package version to the upstream branch of PR #19290.

---

# MoE Load Balancer 中文版

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

## SGLang PR Safety

SGLang PR #19290 是独立 PR，必须保持可单独 review、测试和合入。这个模块的后续接入实验不能直接修改或 push 到 PR #19290 的真实 branch。

开发规则:

- PR #19290 branch 只保留该 PR 自身需要的 Waterfill 实现，不 import 这个尚未 ready 的 `moe_load_balancer` 模块。
- 模块化接入实验必须使用单独 worktree / branch，例如 `pr-19290-moe-lb-integration`，不要覆盖 `feat/deepep-waterfill-eplb-balance` 或 PR 对应 branch。
- 远端验证时，SGLang integration worktree 只应用实验 patch；模块代码通过独立 package 安装，例如 `pip install -e /path/to/moe_load_balancer --no-deps`。
- 除非明确决定切换 PR 策略，否则不要把 import-package 版本 push 到 PR #19290 的 upstream branch。
