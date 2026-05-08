# SGLang POC Plan

本文档是 Stage 3 SGLang POC 的入口说明。读完后应能明确:

1. 独立 MoE load balancer 模块如何定义。
2. core API 和 SGLang adapter API 如何设计。
3. 哪些逻辑必须保持框架无关，哪些逻辑必须留在 SGLang adapter。
4. 当前 POC 做什么、不做什么，以及如何对照阅读 [sglang-integration.md](sglang-integration.md)。

## One-Page Summary

当前 POC 不是完整三层系统，也不是直接把 PR #19290 的 `--enable-deepep-waterfill` 路径搬成通用接口。它要验证的是:

```text
SGLang EPLB placement metadata
  -> SGLang adapter converts to PlacementSnapshot
  -> framework-independent L2 router consumes token-level logical top-k + placement/rank mapping + optional stage
  -> SGLang adapter writes RoutingDecision back to TopK / DeepEP dispatch layout
```

核心边界:


| 层面           | 当前 POC 定义                            | 说明                                                         |
| ------------ | ------------------------------------ | ---------------------------------------------------------- |
| L1 placement | 只复用 SGLang EPLB                      | 不做新的 placement optimizer，不做 per-layer replica budget       |
| L2 routing   | shared-expert Waterfill + LPLB 切片              | 是否只在 prefill、只在 decode 或两者都运行由 core policy config 控制；某个 SGLang 实验可选择 decode bypass            |
| L3 prefetch  | 不做                                   | 只保留长期架构位置                                                  |
| SGLang 集成    | thin adapter + stage-aware TopK hook | SGLang 负责 framework object、id-space、DeepEP layout、通信       |
| Core module  | framework-independent API            | 不 import SGLang 类型，不直接调用 SGLang process group / all-reduce |


PR #19290 latest head `43fe3d965` 把 Waterfill 定成 **server arg 全局开关 + TopK 自发现 + ModelRunner 注入** 的结构: `--enable-deepep-waterfill` 会自动切到 DeepEP 并强制 shared expert fusion；`TopK.__init__` 读取全局 server args 决定是否进入 Waterfill routed-only TopK；`ModelRunner._prepare_moe_topk()` 在模型加载后构造并注入 `DeepEPWaterfillBalancer`；`TopK.forward_*()` 最后调用 `expand_topk()`。latest head 已删除 `static_rank_load`，static path 改为当前 local routed counts 的 no-all-reduce path。这证明 TopK remap 边界是合适的工程落点，也说明 runtime 对象初始化应留在 SGLang 侧；但 PR 的实现仍是 SGLang/DeepEP-specific、无 stage/context 的全局 feature，不等于当前 POC 的 framework-independent interface。

## Relationship To SGLang Integration Doc

本文档负责回答 "POC 怎么设计"。`sglang-integration.md` 负责回答 "SGLang 代码里具体落在哪里"。


| 本文档章节                      | 对照阅读                                                                   |
| -------------------------- | ---------------------------------------------------------------------- |
| `Module Definition`        | `Recommended POC Interface`                                            |
| `Core Interfaces`          | `ID-Space Contract`、`L1 EPLB Adapter Interface`、`L2 Prefill Interface` |
| `SGLang Adapter Contract`  | `Current EPLB Flow In SGLang`、`EPLB Data Contracts To Reuse`           |
| `SGLang Integration Shape` | `Intrusion Map For Current POC`                                        |
| `Policy Scope`             | `Waterfill PR Intrusion`、`Important Gaps For Our POC`                  |


## Module Definition

当前 POC 按 `core + adapter` 定义模块，即使代码暂时 vendored 到 SGLang 仓库，也要保持这个边界。

### Core Owns

Core 是未来可以抽成独立 package 的部分。它只处理框架中立的数据和策略:

- `PlacementSnapshot`: 供 routing 使用的 placement view。
- `RoutingRequest`: 当前层 logical top-k、weights、optional stage、placement、token mask/count。
- `RoutingDecision`: routing 后的框架中立结果。
- `TokenRouter.route_tokens(...)`: L2 policy 入口。
- L2 policy: `shared_expert_waterfill_only`、`lplb_only`、`lplb_then_shared_expert_waterfill`。

以下内容归 SGLang adapter，不作为 core public API:

- `StandardTopKOutput`。
- `ExpertLocationMetadata` / `ExpertLocationDispatchInfo`。
- SGLang global metadata。
- DeepEP process group、`get_moe_ep_group()`、`moe_expert_parallel_all_reduce()`。
- DeepEP interleaved expert id、shared expert fusion slot 等 SGLang/DeepEP layout。
- `ForwardBatch` 或 `ForwardMode` 原始对象。

也就是说，core API 不应该长成 `route_tokens(StandardTopKOutput, ExpertLocationMetadata, ForwardBatch)`；它应该接收 `RoutingRequest` 这种框架中立对象，再由 SGLang adapter 负责类型转换。

### SGLang Adapter Owns

Adapter 是故意耦合 SGLang 的部分。它不是新的算法 core，而是把 SGLang runtime 对象转换成 core 能理解的数据，并把 core decision 写回 SGLang。

先定义一个本文档里的名字: `SGLangRoutingContext` 是 POC proposed adapter-only dataclass，不是 SGLang 当前已有类型。它会作为新增的 `moe_lb_context` 参数，从 `DeepseekV2MoE.forward_deepep` 传给 `TopK` hook，只携带 optional stage / token mask / policy config 这类 per-forward 控制信息。


| Adapter 子职责                            | SGLang 触点                                                                 | 为什么归 adapter                                                                                       | 当前 POC 行为                                                                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 导入 placement snapshot                  | `ExpertLocationMetadata` / `ExpertLocationDispatchInfo`                   | 这是 SGLang EPLB 的内部 metadata 格式                                                                     | 只读 metadata，转换成 `PlacementSnapshot`                                                                                        |
| 跟随 runtime rebalance                   | `EPLBManager.rebalance()`、`ModelRunner.update_expert_location(...)`       | rebalance 和权重更新是 SGLang runtime 流程                                                                 | 不接管 rebalance；rebalance 后刷新或 lazy read 最新 snapshot                                                                         |
| 捕获 logical top-k                       | `topk.py::_post_process_topk_ids` 中 `topk_ids_logical_to_physical(...)` 前 | core 以 logical expert id 表达模型语义，SGLang remap 后的 id 由 adapter 解释                                    | 在 remap 前调用 adapter；或让 adapter 显式接管 remap                                                                                  |
| 传递 stage/context                       | `DeepseekV2MoE.forward_deepep` -> `self.topk(..., moe_lb_context=...)`    | `ForwardBatch` / `ForwardMode` 是 SGLang 类型，需要先压缩成框架中立 stage/mask                                   | forward 只构造最小 `SGLangRoutingContext`，TopK hook 消费它                                                                         |
| 提供 logical->physical placement         | `ExpertLocationDispatchInfo` / global `ExpertLocationMetadata`            | logical->physical replica 表是 SGLang EPLB runtime state                                             | adapter 在 TopK hook 处构造 `PlacementSnapshot`，与 logical top-k 一起传给 core                                                      |
| 初始化 runtime adapter/router             | `ModelRunner.initialize()` / PR 中的 `_prepare_moe_topk()`                  | 模型遍历、EP rank/world、process group 都是 SGLang runtime 细节                                              | 借鉴 PR 的 prepare/inject 模式；PR 的 `DeepEPWaterfillBalancer` 要拆成 core policy 和 SGLang/DeepEP adapter wrapper 两部分               |
| 做 id-space 转换                          | logical id / SGLang physical id / DeepEP interleaved id                   | 各框架 dispatch id 规则不同                                                                               | adapter 把 `RoutingDecision` 转成 SGLang/DeepEP 需要的 ids                                                                       |
| materialize shared expert route column | DeepEP shared fusion / Waterfill expanded TopK layout                     | core 可以表达 “routed top-k + shared route” 的抽象扩展；“追加第几列、写什么 expert id、weight 怎么填”是 DeepEP TopK 输出格式约定 | adapter 把 core 的 `shared_expert_rank` / shared route materialize 成 SGLang/DeepEP 可 dispatch 的 shared expert column，并保持输出等价 |
| 做 runtime communication                | EP group、`moe_expert_parallel_all_reduce()`                               | process group 和 collective API 是框架 runtime 细节                                                      | dynamic Waterfill 需要全局 rank load 时，在 adapter 内 all-reduce                                                                  |
| 写回 SGLang 对象                           | `StandardTopKOutput`                                                      | core public API 不暴露 SGLang 类型                                                                      | adapter 返回可传给 `self.experts(...)` 的 `StandardTopKOutput`                                                                   |


这里要区分 **semantic expansion** 和 **layout materialization**。`top_k=8 -> routed top-k + shared route` 这个语义扩展可以属于 core；PR 的 DeepEP shared fusion 最终需要的 expanded TopK `[num_tokens, 9]` 是具体输出格式。core 可以回答“这个 token 多一条 shared route，目标 EP rank 是谁、权重是多少”；adapter 再把这个 rank 翻译成 DeepEP 需要的 expert id，并把它作为第 9 列写入 `StandardTopKOutput.topk_ids/topk_weights`。

## Core Interfaces

### PlacementProvider

当前 POC 不定义新的 `PlacementPlanner`。L1 只提供 EPLB-backed placement snapshot。

```python
@dataclass
class PlacementSnapshot:
    layer_id: int
    num_logical_experts: int
    num_physical_experts: int
    logical_to_physical_candidates: Tensor
    logical_to_physical_count: Tensor
    default_physical_for_logical: Optional[Tensor] = None
    physical_to_rank_map: Optional[Tensor] = None


class PlacementProvider:
    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        ...
```

说明:

- `logical_to_physical_candidates`: 每个 logical expert 当前有哪些 physical replicas，shape 类似 `[num_logical_experts, max_replicas]`，padding 用 `-1`。
- `logical_to_physical_count`: 每个 logical expert 的有效 replica 数量，用来忽略 padding。
- `default_physical_for_logical`: 可选。表示不做 LPLB 时，当前 framework baseline 会把 logical expert 派发到哪个 physical expert。SGLang static dispatch 的 `logical_to_rank_dispatch_physical_map` 可以转换成这个字段。
- `physical_to_rank_map`: 可选。表示每个 physical expert 所在 EP rank，shape `[num_physical_experts]`。这是 placement/topology 信息，不是负载信息；它主要用于 LPLB/routed-replica 决策，或在 core/adapter 需要从 physical top-k 派生 rank load 时使用。
- `layer_id` 和 `num_*` 主要用于 cache key、校验和统计；很多情况下可以从 tensor shape 推导，但显式保留能让 adapter contract 更清楚。
- `physical_to_logical_map` 和 `num_local_physical_experts` 不建议放进 core snapshot。它们可以留在 SGLang adapter 内部，用来派生 `physical_to_rank_map` 或 `logical_to_physical_candidates`。
- POC proposed `SGLangRoutingContext` 不携带 logical->physical mapping 这种大 tensor。它只把可用的 per-forward 上下文传给 TopK hook，例如 optional stage、token mask 和 policy config。logical->physical mapping 由 `SGLangPlacementAdapter` 从 `ExpertLocationMetadata` / `ExpertLocationDispatchInfo` 读取并转换成 `PlacementSnapshot`。
- 当前 L2 POC 先不输出 `PlacementPlan`。core placement plan 是否需要写回 `ExpertLocationMetadata` 暂时待定，等待 L1 placement 调研结论。

### TokenRouter

```python
@dataclass
class RoutingRequest:
    layer_id: int
    logical_topk_ids: Tensor
    topk_weights: Tensor
    placement: PlacementSnapshot
    stage: Optional[Literal["prefill", "decode", "mixed", "idle", "speculative"]]
    current_rank_load: Optional[Tensor]
    token_count: Optional[Tensor]
    lb_token_mask: Optional[Tensor]
    routed_scaling_factor: float
    policy: RoutingPolicyConfig


@dataclass
class RoutingLoadView:
    current_rank_load: Optional[Tensor]
    prior_rank_load: Optional[Tensor]


class RoutingLoadProvider:
    def load_view(self, layer_id: int) -> RoutingLoadView:
        ...


@dataclass
class RoutingPolicyConfig:
    name: str
    # None means this policy is not gated by stage.
    enabled_stages: Optional[set[Literal["prefill", "decode", "mixed", "idle", "speculative"]]]


@dataclass
class RoutingDecision:
    routed_physical_topk_ids: Tensor
    topk_weights: Tensor
    shared_expert_rank: Optional[Tensor] = None


class TokenRouter:
    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        ...
```

关键语义:

- `logical_topk_ids` 是 token-level logical top-k ids，不是聚合后的 logical expert 分布。
- LPLB / routed-replica 决策需要 `PlacementSnapshot`。shared-expert Waterfill 本身只需要 per-rank load。latest PR #19290 已删除 `static_rank_load`，因此 PR 对齐的 Waterfill 不再消费 EPLB/history 水位线；static path 用当前 local routed counts，dynamic path 用 all-reduced current effective load。EPLB / history 这类先验水位线仍可由 L2 `RoutingLoadProvider` 作为长持 state 提供给其他 policy。
- `stage` 是 optional runtime context，不是每个算法必须使用的分类轴。某个 policy 是否允许在当前 stage 运行，由 core 级 `RoutingPolicyConfig.enabled_stages` 统一判断；`enabled_stages=None` 表示该 policy 不做 stage gating，并允许 `stage=None`。显式列出 stage 则要求调用方提供已知 stage。
- `lb_token_mask` 用于 mixed batch。如果没有可靠 mask，当前 POC 对 `MIXED` 直接 bypass。
- `routed_physical_topk_ids` 使用 `PlacementSnapshot` 定义的框架中立 physical expert namespace，不是 DeepEP interleaved id。
- `shared_expert_rank` 表达 shared expert route 的目标 EP rank；这是 `top_k=8 -> routed top-k + shared route` 的框架中立语义，SGLang adapter 再把它 materialize 成 DeepEP expanded TopK 的第 9 列。
- `RoutingDecision` 是框架中立结果。SGLang adapter 再把它写成 `StandardTopKOutput`。

L2 中两个 rank-load 来源的区别:


| 字段                 | 含义                                                                                | 什么时候需要                                                    |
| ------------------ | --------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `RoutingRequest.current_rank_load` | 当前 batch / 当前 routing window 里，routed experts 已经落到各 EP rank 的 token load | latest PR shared-expert Waterfill 的直接输入 |
| `RoutingLoadView.prior_rank_load`  | L2 先验负载: EPLB / 历史统计给出的 per-rank 水位线，不代表当前 batch 已经实际 routed 的 token 数 | 不再是 latest PR Waterfill 的必需输入，保留给其他 policy |


`current_rank_load` 和 `prior_rank_load` 不要求同时提供，也不要求来自同一个对象。对齐 latest PR #19290 时，Waterfill 只需要 `current_rank_load`:


| Waterfill 模式          | 使用哪个 load                                         | 语义                                                         |
| --------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| dynamic               | `RoutingRequest.current_rank_load`                                | 根据当前 batch/window 的实际 routed counts 给 shared expert 选 rank |
| static / PR static 对齐 | `RoutingRequest.current_rank_load` | 使用当前 local routed counts，避免每次 all-reduce |


Shared-expert Waterfill 的 rank-load 来源按 policy 分开:

```text
Waterfill-only static:
  baseline EPLB routed physical top-k
  -> adapter counts local routed load
  -> RoutingRequest.current_rank_load
  -> shared-expert Waterfill

Waterfill-only dynamic:
  baseline EPLB routed physical top-k
  -> adapter counts routed load
  -> EP all-reduce routed counts and token counts once
  -> RoutingRequest.current_rank_load
  -> shared-expert Waterfill

LPLB + Waterfill:
  logical_topk_ids / local routed counts
  -> EP all-reduce once before LPLB
  -> global per-rank / per-expert load snapshot
  -> LPLB chooses routed physical top-k
  -> derive post-LPLB current_rank_load
  -> shared-expert Waterfill reuses that rank load
```

因此 `physical_to_rank_map` 不是 Waterfill core 的必需参数。它是从 physical top-k 推导 rank load 的一种通用方式。PR #19290 当前 dynamic path 没有显式传这个字段，因为 SGLang/DeepEP 的 physical expert id 已经按 rank 连续分段。例如 `experts_per_rank = 4` 时:

```text
physical expert id 0..3   -> rank 0
physical expert id 4..7   -> rank 1
physical expert id 8..11  -> rank 2
```

所以 PR 里可以用 `target_rank = expert_id // experts_per_rank` 从 physical id 直接推出 rank，然后统计每个 rank 的 routed load。框架解耦接口不能假设所有 backend 都这样编号；如果 adapter 已经用框架自己的规则算好 `current_rank_load`，Waterfill core 就不需要 `physical_to_rank_map`。

组合策略的通信边界要特别明确: `lplb_then_shared_expert_waterfill` 不应该让 LPLB 做一次 EP group 通信、Waterfill 再做一次动态 all-reduce。全局负载通信应发生在 LPLB 之前，并被 LPLB 和 Waterfill 共享；Waterfill 消费的是 LPLB 决策后的 post-LPLB rank load。

## SGLang Adapter Contract

SGLang adapter 可以有局部 helper，但这些 helper 不是 core API。下面的 `SGLangRoutingContext` 是 POC proposed type，对应传给 `TopK` 的新增参数 `moe_lb_context`。

```python
@dataclass
class SGLangRoutingContext:
    policy: RoutingPolicyConfig
    stage: Optional[Literal["prefill", "decode", "mixed", "idle", "speculative"]]
    lb_token_mask: Optional[Tensor]
    num_token_non_padded: Optional[Tensor]


class SGLangPlacementAdapter:
    def snapshot(
        self,
        layer_id: int,
        metadata: ExpertLocationMetadata,
    ) -> PlacementSnapshot:
        ...


class SGLangRoutingAdapter:
    def route_topk(
        self,
        logical_topk_output: StandardTopKOutput,
        placement: PlacementSnapshot,
        context: SGLangRoutingContext,
    ) -> StandardTopKOutput:
        ...
```

Adapter 内部步骤:

1. 从 `ExpertLocationMetadata` / `ExpertLocationDispatchInfo` 拿 per-layer placement view。
2. 转成 `PlacementSnapshot`，包含 logical->physical replicas 和 physical->rank view。
3. 在 SGLang remap 前拿到 logical top-k ids。
4. 根据 `SGLangRoutingContext` 判断是否进入 core `route_tokens(...)`。
5. 把 core decision 转成 SGLang physical id / DeepEP interleaved id。
6. 处理 shared expert slot、padding mask、`num_token_non_padded`。
7. 必要时在 adapter 内做 EP all-reduce 或读取 SGLang runtime group；组合策略下这次通信发生在 LPLB 前，并同时服务 LPLB 和 Waterfill。
8. 返回 SGLang 可直接传给 `self.experts(...)` 的 `StandardTopKOutput`。

`SGLangRoutingContext` 和 `PlacementSnapshot` 是两条不同输入:

```text
DeepseekV2MoE.forward_deepep
  -> builds SGLangRoutingContext(optional stage / token mask / policy config)

TopK hook / SGLang adapter
  -> reads ExpertLocationMetadata or ExpertLocationDispatchInfo
  -> builds PlacementSnapshot(logical->physical placement)

SGLangRoutingAdapter.route_topk(...)
  -> combines context + placement + logical top-k
  -> calls core TokenRouter
```

这样做是为了不让每次 forward 都把 placement mapping 当作 context 参数层层传下去。context 是 per-forward 控制信息；placement 是 runtime metadata，可以由 adapter 在 TopK hook 处读取或缓存。

## SGLang Integration Shape

推荐方案是 **forward 传控制信息，TopK hook 做 routing**。这不是一个已经存在的 SGLang hook 名字，而是 POC 的接入方式。

下面只列出 online forward data path 的两个关键触点，不是完整模块清单。完整 POC 仍包括 core policy、SGLang adapter、`server_args`、`ModelRunner` prepare/inject、placement refresh、EP communication 和 writeback 等部分；这里只强调为什么 context 从 `forward_deepep` 来、routing call 落在 TopK remap 边界。

```text
DeepseekV2MoE.forward_deepep
  能看到 forward_batch.forward_mode，所以负责构造 moe_lb_context
  不直接做 routing

TopK / _post_process_topk_ids
  能看到 remap 前的 logical_topk_ids，所以负责调用 adapter/core
  需要消费 moe_lb_context；core policy config 统一判断当前 optional stage 是否允许该 policy 运行
```

这里的 “stage-aware TopK hook” 指的是 TopK 里的 adapter 调用点带有 `moe_lb_context`，在有 stage 信息时能区分 prefill / decode / mixed。没有 stage 信息时，只有 `enabled_stages=None` 的 policy 可以运行；这让 Waterfill 既可以 stage-agnostic，也可以被配置成只在某些 stage 启用。

PR #19290 最终实现把 shared-expert Waterfill 拆成 **ModelRunner prepare** 和 **TopK apply**:

```text
ServerArgs._handle_a2a_moe
  --enable-deepep-waterfill auto-sets moe_a2a_backend=deepep
  enforce shared expert fusion
  support deepep_mode auto / normal / low_latency

TopK.__init__
  if num_fused_shared_experts > 0:
      read get_global_server_args().enable_deepep_waterfill
  if waterfill is enabled:
      reduce top_k to routed-only columns
      clear num_fused_shared_experts from the base TopK config
      force STANDARD output
      keep deepep_waterfill_balancer = None

ModelRunner.initialize
  load_model()
  _prepare_moe_topk()
    traverse TopK modules
    construct DeepEPWaterfillBalancer with EP rank/world + layer_id
    inject it into TopK.deepep_waterfill_balancer

TopK.forward / empty_topk_output
  select routed experts
  require balancer to be prepared when waterfill is enabled
  apply DeepEPWaterfillBalancer.expand_topk(...)
```

当前 POC 沿用这个 TopK-level 落点，并借鉴 `ModelRunner` prepare/inject 模式初始化 SGLang adapter。但需要把 PR 的 global-flag-driven Waterfill path 扩展成更通用的 adapter hook:

PR 里不是所有改动都进入 generic TopK adapter hook。它们分成三类:

- **Core**: 放框架无关的决策逻辑，例如 Waterfill 选择哪个 rank。
- **Adapter**: 放 SGLang/DeepEP 对象、layout、id-space、通信和 writeback。
- **SGLang integration glue/config/runtime**: 放 SGLang 自己的开关、合法性检查、模型遍历、TopK 接线和生命周期管理。这不是“什么都不放”，而是不进入我们抽象模块的 public core/adapter API。

| PR 改动点 | POC 归属 | 怎么处理 |
|---|---|---|
| Waterfill rank-selection policy | Core | 从 `DeepEPWaterfillBalancer` 中抽出 framework-independent policy: 输入 rank load / candidate ranks，输出 shared expert 应落在哪个 rank |
| `DeepEPWaterfillBalancer` / `deepep_waterfill.py` 中的 SGLang/DeepEP 包装 | Adapter | 保留 `StandardTopKOutput`、DeepEP physical id layout、shared route column materialization、empty output handling 等框架相关部分 |
| current routed rank load | Adapter | latest PR 的 static path 统计 local routed counts；dynamic path all-reduce current routed counts/token counts；不再依赖 `ExpertLocationMetadata.rank_load` |
| EP all-reduce | Adapter | 通信由 SGLang adapter 做；`LPLB + Waterfill` 时在 LPLB 前做一次并共享结果，Waterfill 不再重复通信 |
| `ModelRunner._prepare_moe_topk()` 注入 balancer | SGLang integration glue/runtime | 复用 prepare/inject 生命周期，但注入 generic SGLang routing adapter/router |
| `server_args.py` backend / shared-fusion guard | SGLang integration glue/config | 新 POC 开关复用这些合法性检查；backend、fusion、`deepep_mode` 不进入 core |
| `TopK.__init__` 读取 global `enable_deepep_waterfill` | SGLang integration glue / TopK wiring | 不把 global flag 当成 core interface；只保留“base TopK 输出 routed-only columns”的 shape 约束，并由 POC hook/context 控制是否 route |

下面的伪代码同样只是 runtime forward 调用链示意，用来说明 `moe_lb_context` 如何进入 TopK，以及 TopK 如何在 remap 前调用 adapter。它不覆盖 config、prepare/inject、placement refresh、communication 等其他接入点。

```text
DeepseekV2MoE.forward_deepep
  router_logits = self.gate(...)
  context = build_sglang_lb_context(forward_batch)
  topk_output = self.topk(
      hidden_states,
      router_logits,
      num_token_non_padded=forward_batch.num_token_non_padded,
      expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id),
      moe_lb_context=context,
  )
  final_hidden_states = self.experts(hidden_states, topk_output)

TopK / _post_process_topk_ids
  logical_topk_ids are still available
  placement = placement_adapter.snapshot(
      layer_id,
      expert_location_dispatch_info or global ExpertLocationMetadata,
  )
  if context allows L2:
      adapter.route_topk(logical_topk_output, placement, context)
  else:
      baseline topk_ids_logical_to_physical(...)
```

这里的 `moe_lb_context` 是 POC 需要新增的显式接口，不是 PR #19290 已有参数。当前 PR 的 Waterfill 不传 batch/stage context；`TopK.forward_*()` 只收到 `num_token_non_padded` 和 `expert_location_dispatch_info`。SGLang 里虽然有 `get_forward_context()`，但它只在 piecewise CUDA graph 路径由 `set_forward_context(...)` 设置，不覆盖普通 forward，也不是 TopK 的稳定 public contract。当前 POC 不应让 TopK 隐式从这个全局 context 拉 `ForwardBatch`。

注意这里的拆分边界: PR 里的 `DeepEPWaterfillBalancer` 同时包含核心 Waterfill 决策和 SGLang/DeepEP 运行时 glue。POC 应把它拆成两部分:


| 部分                    | 归属      | 内容                                                                                                                                                                                                     |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Waterfill core policy | core    | 根据 request-local `current_rank_load`、可选 `prior_rank_load`、candidate ranks 和 local preference，选择每个 token 的 shared expert target rank                                                                             |
| DeepEP/SGLang wrapper | adapter | 处理 `StandardTopKOutput`、SGLang placement metadata、local/global rank-load 统计、EP all-reduce、DeepEP interleaved id、把 core 的 shared route materialize 成 DeepEP expanded TopK 第 9 列，以及注入到 `TopK` 的 runtime object |


结论: Waterfill/Balance 的框架无关决策属于 core；PR 中和 SGLang/DeepEP 对象、通信、layout 绑定的包装部分属于 adapter。

### Why Not Reuse PR TopK Waterfill Path Directly

PR #19290 最终路径是:

```text
--enable-deepep-waterfill
  -> TopK reads global server args
  -> ModelRunner injects DeepEPWaterfillBalancer
  -> TopK.forward applies expand_topk()
```

这适合作为 SGLang 内部 feature 的实现方式，但不适合作为当前 POC 的最终接口，因为:

- 它由全局 server arg 驱动，仍然没有 `forward_mode` / stage 输入。
- decode、idle、speculative、mixed traffic 都可能进入 Waterfill path。
- 它通过 `ModelRunner` 给 `TopK` 注入 `DeepEPWaterfillBalancer`，仍依赖 SGLang env、`StandardTopKOutput`、global metadata、EP group、all-reduce 和 DeepEP layout。
- 它实现的是 PR-specific shared-expert Waterfill，不是通用的框架解耦 routing API。
- 它受 DeepEP/shared-fusion layout、`num_fused_shared_experts > 0`、STANDARD TopK output 和 DeepEP dispatch id 规则约束；这些属于 SGLang/DeepEP adapter API，不属于 core policy API。

## Policy Scope

当前 L2 policy scope:


| Policy                              | 行为                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `disabled`                          | 直通，不修改 TopK                                                                                            |
| `shared_expert_waterfill_only`      | 对齐 PR #19290: routed TopK/EPLB remap 保持不变，只把 shared expert 作为额外 dispatch slot 分配到 least-loaded EP rank |
| `lplb_only`                         | 用 LPLB 做 per-batch token 重分配                                                                           |
| `lplb_then_shared_expert_waterfill` | 先 LPLB，再追加 PR 对齐的 shared expert Waterfill                                                              |


Waterfill 语义必须写清楚:

- PR #19290 的 Waterfill 是 **shared-expert Waterfill**。
- 它不在 routed expert 的 EPLB replicas 之间重新选副本。
- 如果未来要做 routed expert replica 贪心选择，应单独命名为 `routed_replica_waterfill`，并单独评测。

## What We Do

当前 POC 做:

- 复用 SGLang EPLB placement metadata / init / rebalance / weight update。
- 定义 framework-independent `PlacementSnapshot`、`RoutingRequest`、`RoutingDecision`。
- 实现 SGLang placement adapter，把 `ExpertLocationMetadata` 转成 `PlacementSnapshot`。
- 借鉴 PR 的 `ModelRunner` prepare/inject 模式初始化 SGLang adapter/router。
- 实现 stage-aware TopK hook，在 logical id remap 前调用 SGLang routing adapter。
- 对齐 PR #19290 的 shared-expert Waterfill 语义。
- 保留 LPLB policy facade 和组合实验入口。
- 验证 logits parity、routing 开销、load balance 和端到端吞吐。

## What We Do Not Do

当前 POC 不做:

- 新的 L1 placement optimizer。
- 新的 per-layer replica budget。
- L3 prediction / expert prefetch。
- Metro。
- TensorRT-LLM / vLLM 生产级 adapter。
- 修改 router top-k 逻辑 expert。
- 让 core 的公开接口直接接收或返回 SGLang 内部类型，例如 `StandardTopKOutput`、`ExpertLocationMetadata`、`ForwardBatch`。这些类型只能出现在 SGLang adapter 内。

## Pending Decisions

- core placement plan 是否写回 SGLang `ExpertLocationMetadata`: 待 L1 placement / EPLB 调研结果确定。当前 L2 POC 只读取 SGLang EPLB metadata，不在这个文档里提前定 writeback API。
- LPLB 对 SGLang 的完整代码侵入分析: 需要做，但当前版本暂未展开；后续由同事补充具体 SGLang 改动点。

## Correctness Rules

- `disabled` 模式必须完全等价于 baseline。
- router 的 logical top-k expert 不变。
- shared-expert Waterfill 只追加等价的 shared expert dispatch slot。
- LPLB 或 future routed-replica policy 如果改 routed expert placement，只能在同一 logical expert 的 physical replicas 内选择。
- `enabled_stages` 受限时，不匹配的 stage 默认 bypass；`enabled_stages=None` 的 policy 不按 stage 限制。
- `MIXED` batch 没有可靠 `lb_token_mask` 时默认 bypass。
- core 不处理 padding / interleaved id / process group；这些由 adapter 处理。

## Milestones


| 阶段  | 任务                      | 产出                                                                          |
| --- | ----------------------- | --------------------------------------------------------------------------- |
| P0  | 接口与侵入点确认                | core/adapter 边界、ID-space contract、stage-aware hook 方案                       |
| P1  | SGLang adapter skeleton | `PlacementSnapshot` 导入、logical TopK 捕获、routing decision 回写                  |
| P2  | L2 最小切片         | shared-expert Waterfill facade、LPLB facade 占位、stage gating、logits parity                 |
| P3  | EPLB baseline 对齐        | init/rebalance 路径、placement snapshot 刷新、当前 routed rank-load 统计路径 |
| P4  | 组合实验                    | shared-expert Waterfill、LPLB、组合策略、routing 开销                                |
| P5  | 汇总 benchmark            | SGLang POC prototype + benchmark 报告                                         |


## Benchmark Plan

建议覆盖:

- EPLB baseline。
- EPLB + shared-expert Waterfill。
- EPLB + LPLB。
- EPLB + LPLB + shared-expert Waterfill。

明确不覆盖:

- 新 L1 placement optimizer / per-layer replica budget variant。
- Metro / decode routing variant。
- L3 prefetch variant。

模型和负载:

- DeepSeek-V3/R1 规模优先。
- GSM8K / MMLU。
- serving trace。
- prefill-heavy 和 mixed traffic 优先；decode-heavy 只做 regression/bypass 检查。

指标:

- 端到端吞吐。
- per-layer / per-rank load balance。
- TopK/routing 修改开销。
- SGLang adapter 开销，包括 logical/physical id 转换、padding mask、DeepEP interleaved/shared-slot 处理。
- AlltoAll 通信量和通信不均衡。
- logits parity 和精度等价性。

## Risks And Decisions


| 风险                                                                | 当前决策                                                                                          |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| TopK hook 无 stage 导致配置受限 policy 误触发                                   | 使用 stage-aware TopK hook；`forward_deepep` 只传最小 context；core `enabled_stages` 统一 gating                                        |
| PR 最终 ModelRunner/TopK Waterfill path 是 stage-less global feature | 只复用 prepare/inject 模式、TopK-level 落点和 Waterfill kernel 思路；POC 另加 `moe_lb_context` 和 adapter 边界 |
| rank-load source 的 scope/生命周期不一致                                        | core 保留 request-local `current_rank_load` 和 long-lived `prior_rank_load` 两类输入；latest PR Waterfill 只用当前 routed rank load，其他算法仍可通过 provider 使用历史水位线 |
| `forward_deepep` 过度承载 routing                                     | 不在这里做完整 routing；只作为 fallback                                                                  |
| SGLang remap 后 logical id 丢失                                      | 在 `_post_process_topk_ids` 的 `topk_ids_logical_to_physical(...)` 前调用 adapter                  |
| shared-expert Waterfill 和 routed-replica Waterfill 混淆             | 当前只实现 PR 对齐的 shared-expert Waterfill；routed-replica 单独作为 future work                          |
| L1 scope 膨胀                                                       | 当前只复用 EPLB；新 placement optimizer / replica budget 移出 POC                                      |
| LPLB 代码侵入不清楚                                                      | 当前版本只保留 facade 和组合实验入口；完整 SGLang 侵入分析后续由同事补充                                              |


## Deliverables

1. SGLang POC prototype。
2. SGLang adapter / core boundary 说明。
3. SGLang EPLB adapter / baseline 说明。
4. stage-aware TopK hook 设计说明。
5. correctness / logits parity 报告。
6. shared-expert Waterfill + LPLB 组合实验结论。
7. benchmark 报告。
8. 是否进入 Stage 4 产品化的建议。
