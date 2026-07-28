# AichenF EPLB Code Review

## 先说人话：我们到底在 Review 什么？

SGLang 原来自己完成一整套 EPLB：

1. 模型运行时，统计每个专家接收了多少 token；
2. 每隔一段时间，判断是否需要重新调整专家；
3. 根据负载计算新的专家摆放位置；
4. 在多个 GPU 之间汇总数据；
5. 把专家权重真正搬到新的 GPU；
6. 更新 SGLang 内部的模型状态。

现在这组提交想把其中一部分搬到 MoE Load Balancer（下面简称 MLB）。

先说明范围：仓库现有 Stage 3 架构合约仍写着“当前 POC 只读取 SGLang EPLB
placement，不把新的 L1 planner 搬进 MLB”。下面提出的统一 L1 planner/runtime
属于后续 Review 目标；如果采纳，必须先更新 `docs/architecture.md`、
`docs/sglang-integration.md` 和 `docs/poc-plan.md`，不能把它写成当前既定合约。
而当前源码已经加入 `DeepSeekEPLBPlanner`、recorder/provider/runtime，这本身就是
文档与代码的 scope 不一致，需要在 PR 中明确解决。

我们真正需要判断的不是“新增了哪些类”，而是：

> 哪些事情是通用负载均衡能力，应该放到 MLB？哪些事情只有 SGLang
> 才知道怎么做，必须留在 SGLang？

---

## 最清晰的边界

判断标准只有一句话：

> 只有 SGLang 才知道或才能执行的事情，留在 SGLang；拿到普通负载数字后，
> 任何推理框架都能复用的计算，放到 MLB。

### 应该留在 SGLang

| SGLang 负责什么 | 为什么必须留在 SGLang |
|---|---|
| 模型什么时候执行一次 forward | MLB 不参与模型执行 |
| 每个 token 实际去了哪个专家 | 这是 SGLang 执行引擎产生的数据 |
| 什么时候触发重新摆放 | SGLang 掌握请求和迭代生命周期 |
| 多个 GPU 之间如何汇总数据 | SGLang 管理进程组、rank 和通信 |
| 哪个模型持有哪个 MLB 实例、何时创建和销毁 | 这是 SGLang 的生命周期和多模型管理 |
| 把专家权重搬到新的 GPU | 只有 SGLang 持有真实模型和权重 |
| 更新 SGLang 的模型内部状态 | 这是 SGLang 私有数据结构 |

### 适合放到 MLB

| MLB 负责什么 | 为什么适合放到 MLB |
|---|---|
| 根据阈值判断是否值得重新计算 | 这是负载均衡策略 |
| 根据负载和 GPU 拓扑计算新的专家摆放 | 这是 EPLB 的核心算法 |

SGLang 已经有完整的 backend hook、窗口、physical→logical 转换和 distributed all-reduce。
目标接入直接复用这条官方 recorder 路径；MLB 当前的通用 `EPLBRecorder` 虽然不依赖
SGLang，但在 SGLang 生产路径中会形成第二份窗口，因此不采用。

### 两边之间应该传什么？

只传普通数据：

```text
SGLang → MLB
  专家负载数字
  当前专家对应关系
  专家数量、GPU 数量、节点数量
  最近的利用率

MLB → SGLang
  新的专家对应关系
```

不应该把这些东西交给 MLB：

```text
SGLang 的 ModelRunner
SGLang 的 ServerArgs
SGLang 的模型对象
SGLang 的全局 runtime 生命周期
真实专家权重
GPU 通信组
```

---

## 最简单的完整流程

```text
1. SGLang 运行模型
        ↓
2. SGLang recorder 统计并保存最近 N 次数据
        ↓
3. 每 N 次 forward，SGLang EPLBManager 触发 dump
        ↓
4. SGLang 按官方 recorder 语义完成 physical→logical 和 distributed all-reduce
        ↓
5. Adapter 把全局统计转换成 PlacementRequest
        ↓
6. MLB 判断是否值得调整并计算 candidate plan
        ↓
7. SGLang 按照结果搬动专家权重并更新模型
        ↓
8. SGLang 继续运行
```

一句话概括：

> SGLang 负责“看见运行”和“真正修改模型”；MLB 负责“根据数字做决定”。

---

## 这组提交现在放进 MLB 的东西是否合适？

| 当前放进 MLB 的功能 | 是否合适 | Review 判断 |
|---|---:|---|
| DeepSeek EPLB 算法 | 合适 | 这是最应该迁移的通用计算 |
| 保存最近 N 次专家负载 | 代码本身通用，但不适合作为 SGLang 目标生产路径 | 与官方 recorder 形成第二份窗口 |
| 物理专家负载转换成逻辑专家负载 | 代码本身通用，但 SGLang 目标直接复用官方实现 | 官方 dump 已完成转换和通信 |
| 根据利用率决定是否跳过重新摆放 | 合适 | 属于负载均衡策略 |
| 生成新的专家摆放结果 | 合适 | MLB 的核心职责 |
| 在 apply 前保存最新摆放结果 | 不合适 | SGLang 权重尚未生效，会形成第二份超前真值 |
| 同时保留新旧两套统计方式 | 不合适 | 会让调用方不知道应该用哪套 |
| 在 MLB 中保存一个 SGLang 全局 runtime | 不合适 | 不应该存在进程级 SGLang runtime |
| 在 MLB 中解决 SGLang 谁先启动的问题 | 不合适 | SGLang 应在每个模型启动时创建一个 MLB 实例 |

---

## Review 意见 #1：MLB 对 SGLang 暴露了太多入口

**位置**

`src/moe_load_balancer/adapters/sglang/eplb/__init__.py`

**严重程度**

Major

### 用简单的话说

这个文件相当于 MLB 给 SGLang 提供的一扇门。

现在这扇门上放了很多按钮：

- 有两个按钮都用于拿专家负载；
- 有些按钮只供测试使用；
- 有些按钮可以随时替换正在使用的 runtime；
- 真正的 SGLang 生产代码并不需要其中大部分按钮。

这会导致接入者不知道：

> 我到底应该走哪条路径？

### Review 建议

只保留 SGLang 真正需要的稳定入口。下面的 `SGLangAdapter` 是目标概念名称，
当前源码尚未实现，不能当作已有 API：

如果采用下面的 Review 意见 #2 和 #3，让 SGLang 自己保存统一的
`MoELoadBalancer` 实例，那么 EPLB 专用 runtime 可以删除；如果暂时保留，也只应是
不保存数据的薄调用入口。

```python
from moe_load_balancer import MoELoadBalancer
from moe_load_balancer.adapters.sglang import SGLangAdapter

__all__ = ["MoELoadBalancer", "SGLangAdapter"]
```

旧的统计入口应删除或明确标记为兼容旧代码；测试清理函数不应成为公开接口。

### 可直接贴到 GitHub 的 Review 评论

> **Major:** This package exposes multiple ways to obtain the same EPLB load
> data, together with functions that are only needed by tests or that replace
> process-wide state. From the SGLang side, this makes it unclear which path is
> the supported integration.
>
> Could we expose only the stable operation SGLang actually needs and keep the
> old statistics path and test helpers private? If SGLang owns a
> framework-neutral MLB instance as suggested below, the SGLang-specific API
> can be reduced to a thin integration entry point.

---

## Review 意见 #2：SGLang 不应通过 MLB 的全局变量查找运行对象

**位置**

`src/moe_load_balancer/adapters/sglang/eplb/runtime.py:255`

**严重程度**

Major

### 用简单的话说

SGLang 有几个不同位置需要使用同一个 MLB 实例：

- 收集模型运行产生的专家负载；
- 冷启动时计算初始摆放；
- 周期性重新计算摆放。

当前代码为了让这几个位置找到同一个对象，把一个名为
`SGLangEPLBRuntime` 的对象保存在 MLB 的全局变量里。

这相当于：

```text
SGLang 创建一个 MLB 相关对象
        ↓
交给 MLB 替 SGLang保管
        ↓
SGLang 其他地方再来 MLB 查找它
```

这不合理。哪个模型持有哪个 MLB 实例、什么时候创建和销毁，只有 SGLang 知道。

### 应该怎么做？

SGLang 在模型启动时创建一个 MLB 实例，然后显式交给需要它的几个地方：

```text
SGLang ModelRunner
  └── 持有一个 MoELoadBalancer 实例
       ├── EPLB 专家负载统计使用它
       ├── EPLB 冷启动和周期重新摆放使用它
       └── Waterfill 等其他算法也通过它调用
```

负载历史和最新摆放保存在这个 MLB 实例自己的状态组件中，而不是保存在
SGLang 专用 runtime 或模块全局变量中。

可以从 MLB 删除：

```python
_default_runtime
set_default_runtime
get_default_runtime
ensure_default_runtime
clear_default_runtime
```

### 为什么这是一个 Major 问题？

- 谁先调用，谁的配置就永久生效，结果依赖启动顺序；
- 一个进程如果跑两个模型，两个模型会争用同一个全局对象；
- 某处替换 runtime 后，收集数据和读取数据可能使用不同对象；
- SGLang 自己的模型和实例管理问题被藏到 MLB，边界变得模糊。

### 可直接贴到 GitHub 的 Review 评论

> **Major:** SGLang should not use an MLB module-global variable to find a
> process-wide SGLang runtime. SGLang knows which model owns which load
> balancer instance, when that instance is created, and when it should be
> destroyed.
>
> Could SGLang create one framework-neutral `MoELoadBalancer` instance during
> model initialization and pass the same reference to its single SGLang
> adapter? The official SGLang recorder can remain the statistics owner and
> SGLang metadata can remain the authoritative placement owner; the adapter
> translates those views when calling the shared MLB instance. No
> process-global runtime registry is needed.

---

## Review 意见 #3：当前代码绕过了顶层 `MoELoadBalancer`

**位置**

- `src/moe_load_balancer/adapters/sglang/eplb/runtime.py`
- `src/moe_load_balancer/adapters/sglang/waterfill/runtime.py`
- SGLang `ModelRunner._prepare_moe_topk()`

**严重程度**

Major / 核心抽象缺失

### 用简单的话说

仓库的架构文档已经定义了顶层对象：

```python
class MoELoadBalancer:
    def route_tokens(self, request):
        ...

    def predict_and_prefetch(self, request):
        ...
```

这个对象才代表整个 MoE Load Balancer 模块。SGLang 应该创建并持有它，再通过唯一的
SGLang Adapter 调用它。EPLB、Waterfill、LPLB 等只是这个对象内部选择或组合的算法，不应该各自
变成一套顶层 load balancer。

初始化原则必须明确写成：

```text
正确：
ModelRunner 初始化一个 MoELoadBalancer
  ├── EPLBManager 使用同一个实例
  └── 所有 TopK 层使用同一个实例，通过 layer_id 区分

不正确：
EPLB 初始化一个 SGLangEPLBRuntime
每个 TopK 层再初始化一个 WaterfillRuntime
未来每增加一种算法，再增加一种 XxxRuntime
```

因此，SGLang 不应该单独初始化 Waterfill runtime。Waterfill 是
`MoELoadBalancer` 内部流水线中的一个算法步骤；它的配置和需要的每层状态由同一个
MLB 实例内部管理。

但当前实现没有真正的 `MoELoadBalancer` 类。相反，
`SGLangEPLBRuntime` 自己持有：

```text
EPLB 算法
最近的专家负载
重新摆放策略
最新摆放结果
```

这等于为 EPLB 单独造了一个 load balancer，并绕过了整体模块的统一入口。

更明显的是：这个文件在 Phase 1.5 之后已经不 import SGLang，只接受普通 Tensor
和数字。这说明这些 EPLB 状态和操作应该装进顶层 `MoELoadBalancer` 的内部组件，
而不属于 `adapters/sglang`。

### 建议架构

实现架构文档中已经定义的统一实例：

```python
class MoELoadBalancer:
    def __init__(self, config):
        self.l1_controller = ...  # skip gate + placement policies
        self.l2_runtime = ...     # ordered routing policies

    def plan_placement(self, request):
        ...  # 只返回 candidate

    def route_tokens(self, request):
        ...
```

这些方法名是目标概念接口，不是当前源码已有 API；正式实现前应先更新架构合约。

然后由 SGLang 的 `ModelRunner` 持有整个模块的实例：

```python
class ModelRunner:
    def __init__(self, ...):
        self.moe_load_balancer = MoELoadBalancer(config)
```

以后所有负载均衡能力都由同一个 Adapter 调用同一个 Core 实例：

```python
candidate = self.moe_lb_adapter.plan_placement(self.moe_load_balancer)
routing = self.moe_lb_adapter.route_topk(self.moe_load_balancer, topk_output)
```

Core 只保存 policy 配置和算法需要的框架中立中间状态，不保存 SGLang recorder 的第二份窗口，
也不保存一份超前的 committed placement。所谓“一个实例”不表示所有模型共享一个进程级全局对象；
而是每个 SGLang `ModelRunner` 显式持有自己的实例。

```text
SGLang ModelRunner
  ├── SGLang recorder / authoritative metadata
  ├── 一个 SGLang Adapter
  └── 一个 MoELoadBalancer
       ├── L1：EPLB candidate planning
       ├── L2：Waterfill / LPLB routing pipeline
       └── L3：未来的预测和预取
```

### Waterfill 提供的参考

现有 Waterfill 的 SGLang 集成采用了正确的生命周期模式：

```text
ModelRunner._prepare_moe_topk()
  └── 创建对象并注入 TopK

TopK.forward()
  └── 直接调用对象的 expand_topk()
```

应该复用的是“由 SGLang 创建、持有/注入、直接调用”这个模式。当前 Waterfill 仍然是
每个 `TopK` 层一个 `WaterfillRuntime`，所以它也不是最终的统一顶层抽象。整体架构应
再向上收敛为：

```text
ModelRunner 创建一个 MoELoadBalancer
  ├── TopK 调用它的 Waterfill/L2 能力
  └── EPLBManager 调用它的 EPLB/L1 能力
```

### 可直接贴到 GitHub 的 Review 评论

> **Major:** The architecture already defines `MoELoadBalancer` as the
> framework-neutral module-level entry point, but the current implementation
> bypasses it and makes `SGLangEPLBRuntime` own EPLB statistics, policy,
> planner, and placement state. This effectively creates an EPLB-specific
> load-balancer object instead of integrating EPLB as one capability of the
> whole MoE load balancer.
>
> Could `ModelRunner` construct and explicitly hold one `MoELoadBalancer`
> instance, following Waterfill's prepare/inject/direct-call lifecycle?
> The single SGLang adapter can translate the official recorder dump and
> authoritative SGLang metadata when calling that shared instance. EPLB and
> Waterfill are then policies of the same top-level object, without moving a
> second recorder or committed placement into MLB. This also removes the need
> for an EPLB-specific runtime and process-global runtime registry.
>
> Concretely, SGLang should not initialize a separate
> `SGLangEPLBRuntime`, nor should each TopK layer initialize its own
> `WaterfillRuntime`. `ModelRunner` should create one `MoELoadBalancer` per
> model/engine and pass the same reference to EPLBManager and every TopK layer;
> per-layer behavior can be keyed by `layer_id` inside that instance.

---

## Review 意见 #4：负载统计、EP 通信和 SGLang 输出转换应该放在 SGLang Adapter

**位置**

`src/moe_load_balancer/adapters/sglang/waterfill/runtime.py`

**严重程度**

架构边界澄清

### 用简单的话说

Waterfill 每次运行前需要做这些事情：

- 从 SGLang `TopKOutput` 里读 routed expert；
- 统计每个 EP rank 的 routed load；
- 必要时调用 SGLang EP all-reduce；
- 把 `ForwardBatch` 和 TopK 数据整理成普通的 `RoutingRequest`；
- 把 `RoutingDecision` 写回 `StandardTopKOutput` 和 DeepEP layout。

这些都应该放在 SGLang Adapter，因为它们直接依赖 SGLang 对象、通信组和 DeepEP
格式。这里的“SGLang Adapter”是指 MLB 仓库中的
`src/moe_load_balancer/adapters/sglang/`，不是把代码搬进 SGLang 主仓。

SGLang 主仓的 `TopK` 只负责把运行时对象交给 MLB 提供的集成对象，并调用一次
`route_topk(...)`；Waterfill 核心算法只接收普通的 `RoutingRequest`。

从 SGLang 看，正确调用链应该只有一个模块对象：

```text
SGLang TopK
  → MoELoadBalancer.route_topk(...)
      → 内部 Adapter：统计负载 / EP all-reduce / 构造 RoutingRequest
      → 内部 core：根据配置执行单算法或组合流水线
      → 内部 LPLB / Waterfill：依次更新 RoutingDecision
      → 内部 Adapter：写回 SGLang TopK 格式
  → SGLang DeepEP dispatch
```

当前 `WaterfillRuntime` 已经做了大部分 Adapter 工作；问题不是这些代码需要删除，
而是它还自己创建并持有 `WaterfillL2Router`，从而绕过统一的 `MoELoadBalancer`。
建议把这些 SGLang 专用操作保留为 `MoELoadBalancer` 实例内部的 SGLang Adapter
组件，把算法选择交给同一个实例内部的 core。

这样 SGLang 主仓只需要两处很薄的 glue：

```text
ModelRunner：创建/注入一个 MoELoadBalancer 实例
TopK：调用 moe_load_balancer.route_topk(...)
```

count kernel、EP all-reduce、`RoutingRequest` 构造和 DeepEP layout 写回全部留在 MLB
仓库，不能散落到 SGLang 的 `ModelRunner`、`TopK` 和 MoE layer。

Adapter 可以是薄对象，也可以是无状态 helper。即使它为了性能复用一个 GPU count
buffer，也不应该拥有 Waterfill/EPLB 算法状态或进程级 registry。

### 可直接贴到 GitHub 的 Review 评论

> The routed-load collection, SGLang EP all-reduce, `ForwardBatch`/TopK
> conversion, and DeepEP output materialization belong in the SGLang adapter
> inside the `moe_load_balancer` package, because they depend on SGLang runtime
> objects and layout. They should not be spread across SGLang's ModelRunner,
> TopK, and MoE layer. SGLang itself should only construct/inject the integration
> object and make one thin `route_topk(...)` call.
>
> The current `WaterfillRuntime` already performs most of this adapter work,
> but it also constructs and owns `WaterfillL2Router`, bypassing the planned
> unified `MoELoadBalancer` instance. Could SGLang hold one `MoELoadBalancer`
> and call a single `route_topk(...)` entry point, with SGLang conversion,
> internal policy selection, Waterfill execution, and output materialization
> all composed inside that same instance? The framework-neutral
> `route_tokens(request)` contract can remain an internal core boundary rather
> than another SGLang-facing entry point.

---

## Review 意见 #5：不能只根据 policy name 选择一个 Router，必须支持有顺序的算法组合

**位置**

整体 `MoELoadBalancer` L2 设计；当前单算法入口分散在：

- `adapters/sglang/lpcf/runtime.py`
- `adapters/sglang/waterfill/runtime.py`

**严重程度**

Major / 顶层组合能力缺失

### 用简单的话说

负载均衡配置不一定只运行一个算法。例如：

```text
LPLB → Waterfill
```

这里不能从两个 Router 中任选一个，因为两个算法做的是不同事情：

- LPLB 先在同一 logical expert 的 replicas 之间重选 routed physical expert；
- Waterfill 再根据 LPLB 之后的 routed rank load，安排 shared expert 的目标 rank。

所以 `MoELoadBalancer` 应该根据配置组织一条有顺序的算法流水线：

```text
waterfill-only   = [Waterfill]
lplb-only        = [LPLB]
lplb+waterfill   = [LPLB, Waterfill]
```

组合调用链：

```text
Adapter 统计输入并做一次 EP all-reduce
  → LPLB 修改 routed physical top-k
  → 产生/推导 post-LPLB routed rank load
  → Waterfill 消费 post-LPLB rank load，追加 shared expert route
  → Adapter 一次性写回最终 SGLang TopK 输出
```

两个关键约束：

1. 顺序不能颠倒。Waterfill 必须看到 LPLB 之后的 routed load。
2. 不能让 LPLB 和 Waterfill 分别做一次 all-reduce。组合路径只通信一次，共享中间数据。

当前一个 runtime 持有一个 router、调用一次就立即返回的形状无法表达这个组合。组合状态
应该由顶层 `MoELoadBalancer` 管理，而不是由 Adapter 或 SGLang 拼接。

### 可直接贴到 GitHub 的 Review 评论

> **Major:** The top-level L2 design cannot assume that exactly one router is
> selected by `policy.name`. We need to support ordered compositions such as
> LPLB followed by shared-expert Waterfill. LPLB first changes the routed
> physical top-k and produces the post-LPLB rank load; Waterfill must then
> consume that updated load before appending the shared-expert route.
>
> Could `MoELoadBalancer` own an ordered routing pipeline rather than a
> single-router lookup? The combined path should perform the SGLang EP
> communication once before LPLB, pass the post-LPLB rank load to Waterfill,
> and materialize the final SGLang output only once after all configured
> algorithms have run.

---

## Review 意见 #6：Adapter/Runtime 必须按框架复用，不能按“框架 × 算法”展开

**位置**

当前 SGLang adapter 已经按算法出现多套 runtime：

- `adapters/sglang/eplb/runtime.py`
- `adapters/sglang/waterfill/runtime.py`
- `adapters/sglang/lpcf/runtime.py`

**严重程度**

Major / 违背 M + N 扩展目标

### 用简单的话说

如果每个算法在每个框架下都实现一套 runtime：

```text
SGLang × EPLB
SGLang × Waterfill
SGLang × LP-CF
vLLM × EPLB
vLLM × Waterfill
vLLM × LP-CF
...
```

开发量就是 M×N。当前代码已经出现这个趋势：Waterfill 和 LP-CF runtime 都重复处理
stage、负载统计、框架通信、`RoutingRequest` 构造和 `StandardTopKOutput` 写回。

目标应该是：

```text
M 个框架无关算法：
  policies/l1/eplb
  policies/l2/lplb
  policies/l2/lpcf
  policies/l2/waterfill

N 个框架 Adapter：
  adapters/sglang
  adapters/vllm
  adapters/tensorrt_llm

1 个框架无关 Orchestrator：
  MoELoadBalancer
```

因此每个框架只实现一套通用 Adapter/Runtime。执行时每个模型或 engine 创建一个
`MoELoadBalancer` 实例并注入对应框架 Adapter；这里的“一个实例”不是进程全局单例。

一个框架 Adapter 可以在内部拆成 L1/L2 通用能力，但不能按算法拆 runtime：

```python
class FrameworkAdapter:
    # L1 边界
    def collect_placement_stats(self, ...): ...
    def materialize_placement_plan(self, plan): ...  # 纯转换，不调用 Updater

    # L2 边界
    def collect_routing_observation(self, requirements, ...): ...
    def materialize_routing_decision(self, decision): ...
```

调用方向也要明确：

```text
SGLang Glue
  → SGLang Adapter（读取/转换框架数据）
  → MoELoadBalancer Core（编排 L1/L2 policies）
  → SGLang Adapter（把结果转换回框架格式）
  → SGLang Glue
```

L1 最后一步由 SGLang EPLBManager 主动调用自己的 Updater；Adapter 只转换 candidate，
不“应用”模型权重。L2 的最终 TopK/DeepEP 格式可以由 Adapter materialize。
MLB Core 不 import、查找或持有 SGLang Adapter。

### Adapter 不能通过算法名分支

仅仅把多个 `XxxRuntime` 合并成一个大文件还不够。如果 SGLang Adapter 内部仍然写：

```python
if policy == "waterfill":
    ...
elif policy == "lplb":
    ...
```

新增算法时仍然要修改每个框架，开发量还是趋向 M×N。

每个算法应该声明自己需要哪些通用数据，例如：

```text
Waterfill 需要：routed rank load、active token count
LPLB 需要：logical top-k、placement、global expert load
LP-CF 需要：logical top-k、placement、global logical count
```

这是一个尚需设计的概念接口，不是当前已有 API。依赖方向应是 Adapter 先向
Core pipeline 查询通用数据需求，再按数据类型收集一次：

```text
pipeline.required_observations()
  → SGLangAdapter.collect_observations(requirements)
  → canonical RoutingState
  → L2 pipeline
```

这样 Adapter 只知道“rank load”“placement”“logical top-k”等通用概念，不知道
Waterfill/LPLB/LP-CF 的算法细节。

### L1 结果如何自然进入 L2

L1 和 L2 使用同一个 `MoELoadBalancer` 实例和内部状态：

```text
L1：
SGLang EPLBManager 触发官方 recorder dump
  → Adapter 转成 PlacementRequest
  → Orchestrator 调 L1 policy
  → SGLang EPLBManager 调自己的 Updater
  → SGLang metadata 在 apply 成功后成为唯一真值

L2：
框架调用同一个实例的 route_topk(...)
  → Adapter 收集本 batch 的普通 routing observation
  → Adapter 从 SGLang 当前 metadata 读取对应 layer 的 PlacementSnapshot
  → 依次执行 LPLB / LP-CF / Waterfill
  → Adapter 一次性写回最终结果
```

框架仍然在启动/周期调整时触发 L1、在每次 forward 时触发 L2。L2 不读取 MLB
提前保存的 candidate；它通过同一个 SGLang Adapter 读取已经生效的 SGLang metadata。

### 代码目录建议

```text
moe_load_balancer/
├── core/
│   ├── orchestrator.py        # MoELoadBalancer
│   └── state.py               # policy / routing 中间状态；不复制 SGLang committed placement
├── policies/
│   ├── l1/eplb/
│   └── l2/
│       ├── lplb.py
│       ├── lpcf.py
│       └── waterfill.py
└── adapters/
    ├── sglang/
    │   ├── adapter.py         # 一套 SGLang Adapter
    │   ├── load_kernels.py
    │   └── materialize.py
    ├── vllm/
    └── tensorrt_llm/
```

LPLB、LP-CF、Waterfill 共用同一个 SGLang L2 Adapter，但算法本身仍在
`policies/l2/`，不能搬进 Adapter。

### 可直接贴到 GitHub 的 Review 评论

> **Major:** The current layout is starting to create one runtime per
> framework/algorithm pair (`SGLangEPLBRuntime`, `WaterfillRuntime`,
> `LPCFRuntime`). This scales as M×N as more algorithms and frameworks are
> added, while the project architecture promises M+N integration cost.
>
> Could we keep one algorithm-independent adapter/runtime implementation per
> framework and one framework-independent `MoELoadBalancer` orchestrator?
> Each L1/L2 policy should declare the canonical observations it requires.
> The SGLang adapter can query the core pipeline for the union of those
> requirements, collect them once, call the configured policy pipeline, and
> materialize the final result once. LPLB, LP-CF, and Waterfill remain framework-neutral
> policies under `policies/l2/`; they should not each own an SGLang runtime.
>
> L1 and L2 should share the same top-level instance, but SGLang remains the
> authoritative placement owner. After SGLang applies an L1 plan, the single
> SGLang adapter exposes the current metadata as a canonical snapshot; L2
> consumes that applied snapshot, never an unpublished MLB candidate.

---

## Review 意见 #8：新 Placement 在权重搬运成功前就对 L2 可见

**位置**

`src/moe_load_balancer/adapters/sglang/eplb/runtime.py:244-251`

**严重程度：Blocker / Major**

当前 `compute_placement()` 在返回候选布局时立即调用：

```python
self._provider.update_with(plan)
```

但此时 SGLang 尚未根据这个 plan 搬运专家权重，也没有更新 authoritative
`ExpertLocationMetadata`。如果 L2 从 provider 读取新布局，它可能把 token 路由到尚未装好正确
权重的物理槽；如果搬运失败，provider 仍然保存错误的新版本。

正确顺序必须是：

```text
MLB plan candidate
  → Adapter 只做数据结构转换
  → SGLang 搬权重并更新 metadata
  → apply success
  → L2 通过 Adapter 读取 SGLang 已生效 metadata
```

### 可直接贴到 GitHub 的 Review 评论

> **Major:** `compute_placement()` publishes the new plan to
> `EPLBPlacementProvider` before SGLang has moved the expert weights or updated
> its authoritative metadata. An L2 reader can therefore observe a placement
> that is not yet materialized, and a failed apply leaves the provider ahead
> of the model.
>
> Please keep planning separate from authoritative state: `plan()` should
> return an unpublished candidate, SGLang should apply it through its own
> updater, and L2 should read the applied SGLang metadata through the adapter.
> MLB should not keep a second committed placement for this production path.

---

## Review 意见 #9：统计数据是否已经跨 rank 汇总，当前契约自相矛盾

**位置**

- `src/moe_load_balancer/adapters/sglang/eplb/stats_bridge.py:64-83`
- `src/moe_load_balancer/adapters/sglang/eplb/runtime.py:108-121`

**严重程度：Major**

`SGLangStatsBridge` 注释说传入的 `global_physical_count` 已经跨 EP ranks
all-reduce；`SGLangEPLBRuntime.drain_stats()` 又把输出定义为 per-rank view，要求调用方再
all-reduce。

官方 SGLang 的真实顺序是：

```text
每个 rank 取得使用全局 physical expert id 表示的本地计数
  → physical count 转 logical count
  → _StatAccumulator.dump() 中 all_reduce logical count
```

这里的 `global_physical_count` 主要表示“全局 physical-id 的 tensor shape”，不是“已经进行
全局通信”。如果契约不改清楚，接入方可能漏做或重复做通信。

目标 SGLang 生产路径不应修补 Bridge 后继续维护第二套窗口；应直接保留
`SGLangStatsCollector`：它调用官方 `dump_record(output_mode="object")`，取得已经完成
physical→logical 和官方 distributed all-reduce 的 `logical_count`，然后只转换成
`PlacementStats`。Adapter 不应再次 all-reduce。

---

## Review 意见 #10：公开算法枚举与实际实现能力不一致

**位置**

- `src/moe_load_balancer/policies/l1/eplb/algorithms/deepseek_vec.py`
- `src/moe_load_balancer/policies/l1/eplb/algorithms/elasticity_aware.py`

**严重程度：Major；Elastic EP 场景为 Blocker**

MLB 保留了 SGLang 的六个 EPLB algorithm 名字，但当前真正完整的只有
`deepseek` 和 `deepseek_hierarchical`：

| 算法 | 当前 MLB 状态 |
|---|---|
| `deepseek` | 完整 |
| `deepseek_hierarchical` | 完整 |
| `deepseek_vec*` | stub，调用即抛 `NotImplementedError` |
| `elasticity_aware*` | stub，调用即抛 `NotImplementedError` |

官方 SGLang 在开启 Elastic EP 时会把 `auto` 自动解析成 `elasticity_aware`。因此这不是一个
很少使用的显式选项：正常配置就可能进入 stub。

建议本 PR 完整迁入这两个算法并做 upstream 等价测试；如果暂时不支持，就不能把它们作为可用
capability 暴露，并应在 Adapter 初始化时 fail fast，而不是运行到第一次 rebalance 才失败。

---

## Review 意见 #11：SGLang 应通过一个 Request class 向 MLB 提供参数

**位置**

- `src/moe_load_balancer/adapters/sglang/eplb/runtime.py:187-243`
- `src/moe_load_balancer/core/types.py:259-281`

**严重程度：Major / 公共接口设计**

当前 `compute_placement()` 的外部调用接口要求 SGLang 分散地传入：

```python
compute_placement(
    logical_count=...,
    num_physical_experts=...,
    num_local_physical_experts=...,
    num_groups=...,
    num_nodes=...,
    algorithm=...,
    active_ranks=...,
)
```

参数继续增加后，每个调用位置都要知道完整参数列表，也很容易漏传、错传或出现不同入口使用不同
默认值。虽然函数内部最后会构造 `PlacementRequest`，但这个统一对象没有成为 SGLang→MLB 的
正式公共边界。

建议由唯一的 SGLang Adapter 从 `server_args`、model config、placement metadata 和统计结果中
抽取普通数据，直接构造仓库已经定义的框架中立 `PlacementRequest`，然后调用目标顶层入口：

```python
request = PlacementRequest(
    stats=PlacementStats(logical_count=global_logical_count),
    policy=PlacementPolicyConfig(
        name="eplb",
        metadata={
            "algorithm": algorithm,
            "num_physical_experts": num_physical_experts,
            "num_local_physical_experts": num_local_physical_experts,
            "num_groups": num_groups,
            "num_nodes": num_nodes,
        },
    ),
    current=current_snapshot,
)
plan = moe_load_balancer.plan_placement(request)  # 目标概念名，当前尚未实现
```

这里优先复用当前合约：layout scalar 按 `PlacementPolicyConfig.metadata` 的既定语义传递，
负载 Tensor 放在 `PlacementStats`，不能臆造当前不存在的 `PlacementTopology` 字段。
SGLang Adapter 负责创建 request，但不能把
`ServerArgs`、`ModelRunner`、process group 或模型对象直接放进去。

同样的规则也适用于 L2：SGLang Adapter 构造一个 `RoutingRequest`，而不是给 Waterfill、LPLB
分别传一套零散参数。

### 可直接贴到 GitHub 的 Review 评论

> **Major:** The SGLang-to-MLB boundary still exposes a long list of scalar and
> tensor arguments on `compute_placement()`, even though the method immediately
> repacks them into a `PlacementRequest`. As the integration grows, every
> SGLang call site will have to duplicate this parameter assembly and default
> handling.
>
> Could the SGLang adapter construct one typed, framework-neutral request and
> pass that request directly to the top-level `MoELoadBalancer`? The request
> should reuse the existing `PlacementRequest` contract: load tensors belong in
> `PlacementStats`, stable layout scalars use `PlacementPolicyConfig.metadata`,
> and SGLang objects/process groups stay outside the core contract. The same
> one-request pattern should be used for L2 through `RoutingRequest`.

---

## Review 范围和链接

- 作者：AichenF
- Author-only 分支：`review/aichenf-only`
- 基线：`f9125d00f30181d2fc84d381a25ae6562b5f2ea2`
- GitHub Diff：
  <https://github.com/xutizhou/moe_load_balancer/compare/f9125d00f30181d2fc84d381a25ae6562b5f2ea2...review/aichenf-only>
- [最简职责边界与流程图](sglang_eplb_runtime_timeline_review.html)
- [整体提交变化说明](eplb_diff_review_212a1df8_to_1772ab76.html)
