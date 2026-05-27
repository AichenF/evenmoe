# MoE Load Balancer

## English

`moe_load_balancer` is a prototype Python package for framework-decoupled
Mixture-of-Experts (MoE) load balancing. It separates algorithm policy code
from framework glue so the same placement/routing ideas can be validated in
SGLang first and later reused by other runtimes such as TensorRT-LLM, vLLM,
NeMo Inference, or Dynamo.

The repo currently focuses on two layers:

- **L1 placement planning**: framework-neutral expert placement plans. The
  current implementation wraps the DeepSeek/SGLang EPLB placement algorithm.
- **L2 token routing**: per-layer, per-batch routing decisions that preserve
  logical router output while choosing physical expert replicas or shared
  expert dispatch targets. Current policies include shared-expert Waterfill
  and LP-CF.

The core rule is accuracy preservation: policies must not change the router's
logical top-k expert choices. They may only choose among equivalent physical
replicas or add framework-supported shared-expert dispatch metadata.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/moe_load_balancer/core/` | Framework-neutral dataclasses and protocols. |
| `src/moe_load_balancer/policies/l1/` | L1 placement planning policies, currently EPLB. |
| `src/moe_load_balancer/policies/l2/` | L2 routing policies, currently Waterfill and LP-CF. |
| `src/moe_load_balancer/adapters/sglang/` | SGLang-specific conversion, runtime glue, and Triton helper kernels. |
| `tests/` | Unit tests for core API contracts and LP-CF behavior. |
| `docs/architecture.md` | Design notes and module boundaries. |
| `docs/lpcf-user-guide.md` | LP-CF usage guide. |
| `docs/h20-pr19290-reproduction-howto.md` | Waterfill reproduction notes for the H20/SGLang PR workflow. |
| `docs/deepep-waterfill-lplb-blog-draft.md` | Blog draft for Waterfill and LPLB work. |

## Install

For local development:

```bash
git clone https://github.com/xutizhou/moe_load_balancer.git
cd moe_load_balancer
pip install -e ".[dev]"
```

For use inside an SGLang worktree or dev container:

```bash
pip install -e /path/to/moe_load_balancer --no-deps
```

The core package has no required runtime dependencies. GPU policy/runtime
modules import `torch` and `triton` only when those modules are imported.

## Run Tests

```bash
pytest
```

Useful narrower runs:

```bash
pytest tests/test_api_contracts.py -v
pytest tests/test_lpcf.py -v
```

Some LP-CF tests are CUDA/Triton parity checks and are skipped automatically
when CUDA is unavailable.

## Use The Core API

Framework-independent callers should depend on `core` types and policy
protocols, not on SGLang classes:

```python
from moe_load_balancer.core.types import RoutingPolicyConfig, RoutingRequest, RoutingStage

request = RoutingRequest(
    layer_id=0,
    stage=RoutingStage.PREFILL,
    logical_topk_ids=topk_ids,
    topk_weights=topk_weights,
    placement=placement_snapshot,
    policy=RoutingPolicyConfig(name="lpcf"),
)

decision = router.route_tokens(request)
```

The adapter is responsible for translating framework-native tensors and
metadata into these neutral request objects, and for writing the returned
decision back into the framework's native dispatch format.

## Use With SGLang

This repo is meant to be imported as a package by an SGLang integration
branch instead of being copied into the SGLang source tree.

In an SGLang dev environment:

```bash
pip install -e /path/to/moe_load_balancer --no-deps
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH
```

SGLang-side glue should stay small and framework-specific:

- read SGLang metadata such as `ExpertLocationMetadata`;
- build `PlacementSnapshot` / `RoutingRequest`;
- call the selected policy/runtime in this package;
- write the `RoutingDecision` back to SGLang top-k, physical expert ids, or
  shared-expert dispatch slots.

Algorithm code that does not depend on SGLang internals should live in
`src/moe_load_balancer/policies/`. SGLang-only conversion code should live in
`src/moe_load_balancer/adapters/sglang/` or in the SGLang integration branch.

## Current Algorithms

### EPLB Placement

`DeepSeekEPLBPlanner` exposes the DeepSeek/SGLang EPLB placement algorithm
through the framework-neutral `PlacementPlanner` protocol. It consumes
logical expert activation statistics and returns a `PlacementPlan` plus raw
metadata that adapters can write back to native placement structures.

### Waterfill Routing

`WaterfillL2Router` implements shared-expert Waterfill routing. It does not
change routed top-k experts. It consumes per-rank routed load and optional
active token counts, then chooses the shared-expert destination rank.

Waterfill is intentionally usable with or without stage gating. A framework
adapter may run it only in prefill, only in decode, in both, or without
passing stage information.

### LP-CF Routing

`LPCFL2Router` is a closed-form L2 routing policy for logical experts with
redundant physical replicas. See `docs/lpcf-user-guide.md` for the SGLang
launch flow, environment variables, tests, and benchmark guidance.

## Development Notes

- Keep framework-neutral API contracts in `core`.
- Keep algorithm logic in `policies`.
- Keep SGLang, DeepEP, process-group, interleaved-id, and native writeback
  logic in adapters or SGLang glue.
- Do not put per-step tensors in policy `metadata`; pass them through
  `RoutingRequest` or a provider interface.
- Stage is optional context. Policies should support ungated execution unless
  a specific algorithm truly requires stage-specific behavior.

---

## 中文版

`moe_load_balancer` 是一个用于 MoE 推理负载均衡的 Python 原型包。它的目标是把
算法 policy 和框架 glue 解耦：先在 SGLang 中开发、验证和跑性能，再尽量复用到
TensorRT-LLM、vLLM、NeMo Inference、Dynamo 等其他运行时。

当前 repo 主要覆盖两层能力：

- **L1 placement planning**：框架中立的专家放置规划。当前实现封装
  DeepSeek/SGLang EPLB placement 算法。
- **L2 token routing**：每层、每 batch 生成 routing decision，在不改变
  router 逻辑 top-k 的前提下，选择物理 expert replica 或 shared expert
  dispatch 目标。当前包含 shared-expert Waterfill 和 LP-CF。

核心约束是精度无损：policy 不能改变 router 选择的逻辑 expert，只能在同一个逻辑
expert 的物理副本之间选择，或生成框架已经支持的 shared-expert dispatch metadata。

## Repo 结构

| 路径 | 作用 |
|---|---|
| `src/moe_load_balancer/core/` | 框架中立 dataclass 和 protocol。 |
| `src/moe_load_balancer/policies/l1/` | L1 placement policy，目前是 EPLB。 |
| `src/moe_load_balancer/policies/l2/` | L2 routing policy，目前是 Waterfill 和 LP-CF。 |
| `src/moe_load_balancer/adapters/sglang/` | SGLang 相关转换、runtime glue 和 Triton helper kernel。 |
| `tests/` | core API contract 和 LP-CF 行为测试。 |
| `docs/architecture.md` | 架构设计和模块边界。 |
| `docs/lpcf-user-guide.md` | LP-CF 使用说明。 |
| `docs/h20-pr19290-reproduction-howto.md` | H20/SGLang Waterfill PR 复现实验说明。 |
| `docs/deepep-waterfill-lplb-blog-draft.md` | Waterfill 和 LPLB blog draft。 |

## 安装

本地开发：

```bash
git clone https://github.com/xutizhou/moe_load_balancer.git
cd moe_load_balancer
pip install -e ".[dev]"
```

在 SGLang worktree 或 dev container 中使用：

```bash
pip install -e /path/to/moe_load_balancer --no-deps
```

core 包本身没有强制 runtime dependency。GPU policy/runtime 只有在导入对应模块时才会
导入 `torch` 和 `triton`。

## 运行测试

```bash
pytest
```

常用的局部测试：

```bash
pytest tests/test_api_contracts.py -v
pytest tests/test_lpcf.py -v
```

部分 LP-CF 测试是 CUDA/Triton parity test；没有 CUDA 时会自动 skip。

## 使用 Core API

框架无关的调用方应该依赖 `core` 类型和 policy protocol，而不是依赖 SGLang 类型：

```python
from moe_load_balancer.core.types import RoutingPolicyConfig, RoutingRequest, RoutingStage

request = RoutingRequest(
    layer_id=0,
    stage=RoutingStage.PREFILL,
    logical_topk_ids=topk_ids,
    topk_weights=topk_weights,
    placement=placement_snapshot,
    policy=RoutingPolicyConfig(name="lpcf"),
)

decision = router.route_tokens(request)
```

adapter 负责把框架原生 tensor 和 metadata 转成这些中立 request object，并把返回的
decision 写回框架自己的 dispatch 格式。

## 在 SGLang 中使用

这个 repo 更适合被 SGLang integration branch 作为 package import，而不是直接复制
到 SGLang 源码树里。

在 SGLang dev 环境中：

```bash
pip install -e /path/to/moe_load_balancer --no-deps
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH
```

SGLang 侧 glue 应该保持小而框架相关：

- 读取 SGLang metadata，例如 `ExpertLocationMetadata`；
- 构造 `PlacementSnapshot` / `RoutingRequest`；
- 调用这个 package 里的 policy/runtime；
- 把 `RoutingDecision` 写回 SGLang top-k、physical expert id 或 shared-expert
  dispatch slot。

不依赖 SGLang 内部结构的算法代码应该放在 `src/moe_load_balancer/policies/`。
SGLang 专属的转换和 writeback 逻辑应该放在
`src/moe_load_balancer/adapters/sglang/` 或 SGLang integration branch 中。

## 当前算法

### EPLB Placement

`DeepSeekEPLBPlanner` 通过框架中立的 `PlacementPlanner` protocol 暴露
DeepSeek/SGLang EPLB placement 算法。它消费逻辑 expert activation statistics，
返回 `PlacementPlan`，同时保留 adapter 写回原生 placement 结构所需的 raw metadata。

### Waterfill Routing

`WaterfillL2Router` 实现 shared-expert Waterfill routing。它不改变 routed top-k
expert，只消费 per-rank routed load 和可选 active token count，然后选择 shared
expert 的目标 rank。

Waterfill 不强制绑定 stage。framework adapter 可以只在 prefill 调用、只在 decode
调用、两者都调用，或者完全不传 stage。

### LP-CF Routing

`LPCFL2Router` 是面向 redundant physical replica 的 closed-form L2 routing
policy。SGLang 启动方式、环境变量、测试和 benchmark 说明见
`docs/lpcf-user-guide.md`。

## 开发约定

- 框架中立 API contract 放在 `core`。
- 算法逻辑放在 `policies`。
- SGLang、DeepEP、process group、interleaved id、原生 writeback 逻辑放在
  adapter 或 SGLang glue。
- 不要把 per-step tensor 放进 policy `metadata`；这类输入应该通过
  `RoutingRequest` 或 provider interface 传递。
- stage 是可选上下文。除非算法本身确实必须区分阶段，否则 policy 应支持不带 stage
  的运行方式。
