# SGLang Integration

本文档记录 P0 阶段的 SGLang 接口确认和侵入点分析。

本轮分析范围:

- 本地 SGLang main: `../sglang`，commit `fdfe53e87`。
- Waterfill PR: <https://github.com/sgl-project/sglang/pull/19290>，fetched as `origin/pr-19290`，commit `4e76f923a`。
  本次相对上一轮 `015d941bd` 新增检查了 `4e76f923a`，该 commit 是当前 PR 最终结构。
- 分析对象: SGLang 现有 EPLB 实现，以及 Waterfill PR 对 SGLang 的代码侵入。

当前版本暂未展开 LPLB 对 SGLang 的完整代码侵入分析。LPLB 仍是后续需要补齐的 L2 prefill 方向；本轮先保留算法接口和组合入口，具体 SGLang 改动点后续由同事补充。

## P0 Goal

P0 的目标不是直接实现完整 load balancer，而是确认:

- L1 如何复用 SGLang 现有 EPLB metadata / rebalance 路径作为 placement source。
- L2 prefill routing 能在 TopK 和 DeepEP dispatch 之间插入到哪里。
- Waterfill 当前 PR 改了哪些 SGLang 代码路径。
- 为了把 Waterfill 变成通用 load balancer 模块，需要把哪些 SGLang 细节留在 adapter。

当前 POC 明确不做:

- L3 prediction/prefetch。
- decode/Metro。

后续待补:

- LPLB 对 SGLang 的完整代码侵入分析。

## Current EPLB Flow In SGLang

EPLB 已经提供了我们需要复用的大部分 L1/L2 基础设施: placement metadata、logical-to-physical dispatch map、expert distribution recorder、periodic rebalance 和 weight update。

| 阶段 | SGLang 位置 | 当前行为 | 对 load balancer 的意义 |
|---|---|---|---|
| CLI/config | `python/sglang/srt/server_args.py` | `--enable-eplb` 自动打开 `expert_distribution_recorder_mode=stat`；如果启用 EPLB 或传入非 trivial `init_expert_location`，默认 `ep_dispatch_algorithm=static`；PR #19290 的 `--enable-deepep-waterfill` 会自动把 `moe_a2a_backend` 改成 `deepep`，并强制 shared expert fusion | POC 新开关应挂在这里；可以借鉴 PR 的自动 backend/fusion 修正，但不要把全局开关当成 core interface |
| 初始化 metadata | `model_executor/model_runner.py::initialize` | 调用 `compute_initial_expert_location_metadata(...)` 并写入 global metadata | POC 直接复用 SGLang EPLB 初始 placement |
| placement 表达 | `eplb/expert_location.py::ExpertLocationMetadata` | 保存 `physical_to_logical_map`、`logical_to_all_physical_map`、`logical_to_rank_dispatch_physical_map` 等 | 我们的 `PlacementView` 应从这里派生，不需要重建 SGLang placement 表达 |
| model 配置 | 各模型 `get_model_config_for_expert_location`，DeepSeek 在 `models/deepseek_v2.py` | 返回 `num_layers`、`num_logical_experts`、`num_groups` | adapter 构造 `PlacementSnapshot` 时的模型维度来源 |
| dispatch info | `eplb/expert_location_dispatch.py::ExpertLocationDispatchInfo.init_new(layer_id)` | 从 global metadata 切出当前层的 logical-to-physical 信息 | L2 每层路由需要复用这里的 per-layer view |
| TopK 后处理 | `layers/moe/topk.py::_post_process_topk_ids` | 调 `topk_ids_logical_to_physical(...)`，把 logical expert id 改成 physical expert id；随后 recorder 记录 `topk_ids` | 这是 L2 最关键的 ID-space 边界；generic router 应在 remap 前拿到 logical ids，或由 adapter 显式关闭默认 remap 后再接管 |
| Waterfill prepare | `model_executor/model_runner.py::_prepare_moe_topk` | PR #19290 在 `load_model()` 后遍历 `TopK` modules，构造 `DeepEPWaterfillBalancer` 并注入到 `TopK.deepep_waterfill_balancer` | runtime 对象准备和 EP rank/world 读取应留在 SGLang adapter，不进入 core |
| DeepSeek MoE | `models/deepseek_v2.py::forward_deepep` | 计算 router logits，调用 `self.topk(..., expert_location_dispatch_info=...)`，再把 `topk_output` 传给 `self.experts(...)`；PR #19290 最终不再给 `TopK` 传 waterfill 参数，而是让 `TopK` 读取 global server args，并由 `ModelRunner._prepare_moe_topk()` 注入 balancer | forward 能拿到 `forward_batch.forward_mode`，但普通 `self.topk(...)` 返回值可能已经是 physical ids；如果在这里插入 generic router，必须确保 adapter 先拿到 logical top-k |
| 统计采集 | `eplb/expert_distribution.py` + `layers/moe/token_dispatcher/deepep.py` | TopK/DeepEP dispatch 后采集 physical count，再转换成 logical count | 当前 POC 只用于理解/记录 EPLB baseline，不实现新 placement optimizer |
| 周期 rebalance | `eplb/eplb_manager.py::rebalance` | dump recorder，调用 `ExpertLocationMetadata.init_by_eplb(...)`，再 `model_runner.update_expert_location(...)` | 当前 POC 继续走 SGLang EPLB 原 rebalance 路径 |
| 权重更新 | `model_executor/model_runner.py::update_expert_location` + `eplb/expert_location_updater.py` | 根据新旧 placement 做 P2P/磁盘加载并更新 global metadata | 当前 L2 POC 不绕过它；core placement plan 是否写回这里待 L1 调研确定 |

## EPLB Data Contracts To Reuse

当前 SGLang 已有 placement contract:

```python
ExpertLocationMetadata:
    physical_to_logical_map              # [num_layers, num_physical_experts]
    logical_to_all_physical_map          # [num_layers, num_logical_experts, max_replicas]
    logical_to_all_physical_map_num_valid
    logical_to_rank_dispatch_physical_map # [num_layers, num_logical_experts], static dispatch only
```

当前 SGLang 已有 per-layer dispatch view:

```python
ExpertLocationDispatchInfo.init_new(layer_id):
    ep_dispatch_algorithm
    partial_logical_to_rank_dispatch_physical_map
    partial_logical_to_all_physical_map
    partial_logical_to_all_physical_map_num_valid
    num_physical_experts
```

POC 里不需要另起一套 placement metadata。建议定义一个薄的 adapter view:

```python
@dataclass
class SGLangPlacementView:
    layer_id: int
    ep_size: int
    ep_rank: int
    num_logical_experts: int
    num_physical_experts: int
    num_local_physical_experts: int
    physical_to_logical_map: Tensor
    physical_to_rank_map: Tensor
    logical_to_all_physical_map: Tensor
    logical_to_all_physical_map_num_valid: Tensor
    logical_to_rank_dispatch_physical_map: Optional[Tensor]
    rank_load: Optional[Tensor] = None
```

`rank_load` 不是 main 上已有字段；它来自 Waterfill PR。虽然它挂在 `ExpertLocationMetadata` 上，adapter 转换时仍应把它拆到 L2 `RoutingRequest.prior_rank_load`，不要放进 `PlacementSnapshot`。

`physical_to_rank_map` 可以由 SGLang adapter 从 global physical expert id 和 `num_local_physical_experts` 推导。它是 placement/topology 信息，不是负载信息；L2 rank-load 字段的语义在 `L2 Prefill Interface` 小节说明。

`SGLangPlacementView` 可以保留 SGLang 字段名，但进入 core 前应拆成 L1 placement 和 L2 routing load:

| SGLang adapter 字段 | 转换目标 | 说明 |
|---|---|---|
| `logical_to_all_physical_map` | `PlacementSnapshot.logical_to_physical_candidates` | 每个 logical expert 的 physical replicas |
| `logical_to_all_physical_map_num_valid` | `PlacementSnapshot.logical_to_physical_count` | replica 有效数量 |
| `physical_to_rank_map` | `PlacementSnapshot.physical_to_rank_map` | physical expert 到 EP rank 的映射，可选 helper |
| `logical_to_rank_dispatch_physical_map` | `PlacementSnapshot.default_physical_for_logical` | SGLang static dispatch 的 baseline remap，可选 |
| `rank_load` | `RoutingRequest.prior_rank_load` | EPLB / 历史 per-rank 水位线，可选；不是 placement 字段 |
| `physical_to_logical_map`、`num_local_physical_experts` | adapter-only | 主要用于派生上述字段，不建议变成 core contract |

## Waterfill PR Intrusion

PR #19290 的目标是 DeepEP Waterfill: 把 shared expert 当作第 9 个 routed expert 送进 DeepEP，并为每个 token 把 shared expert 分配给 least-loaded EP rank。

### Changed Files

| 文件 | 改动 | 侵入性质 |
|---|---|---|
| `server_args.py` | 新增 `enable_deepep_waterfill`；启用时如果 `moe_a2a_backend != deepep` 会自动 override 成 `deepep`，并强制 shared expert fusion；文档声明支持 `deepep_mode=auto`、`normal` 或 `low_latency` | 配置层侵入 |
| `environ.py` | 新增 `SGLANG_DISABLE_STATIC_WATERFILL`；最终文档明确默认优先使用 static EPLB rank-load metadata，设置该 env 后强制 dynamic all-reduce | 调试/策略开关 |
| `eplb/expert_location.py` | 给 `ExpertLocationMetadata` 增加 `rank_load`；在 `init_by_eplb(...)` 中基于 3D `logical_count` 计算 per-rank load，并在 `update(...)` 中按 layer 同步；mapping-only init 无 static rank load | EPLB metadata 侵入 |
| `layers/moe/deepep_waterfill.py` | 新增 Triton kernels 和 `DeepEPWaterfillBalancer`；删除 `WaterfillTopK` wrapper；`update_static_weights()` 只初始化一次 static rank load | 新算法文件，但目前直接 import SGLang env、`StandardTopKOutput` 和 EP all-reduce |
| `layers/moe/topk.py` | 不新增 constructor 参数；当 `num_fused_shared_experts > 0` 时读取 global `server_args.enable_deepep_waterfill`，启用后把 base TopK 改成 routed-only、强制 STANDARD output，并在 forward / empty path 后调用 `expand_topk()`；如果启用但未 prepare 会报错 | Waterfill 主执行点在 TopK，但激活源是全局 server arg，balancer 构造已移出 TopK |
| `model_executor/model_runner.py` | 新增 `_prepare_moe_topk()`；`load_model()` 后遍历 TopK modules，构造并注入 `DeepEPWaterfillBalancer`，同时调用 `update_static_weights()` | runtime prepare/injection 入口 |
| `models/deepseek_v2.py` | 移除模型侧 `_enable_deepep_waterfill` 逻辑；`TopK` 仍按普通 shared-fusion 形态构造；`enforce_shared_experts_fusion` 现在作为通用 fusion override 通过 disable checks | 模型路径侵入继续下降，但仍只覆盖 DeepSeek V2/V3/R1 系列 |
| docs/test | 新增 server args 文档和 unit tests，覆盖强制 shared fusion、backend auto-override 和 `deepep_mode=low_latency` | 配套改动 |

### Waterfill Runtime Shape

当前 main 的 DeepEP shared fusion 会让 TopK 自己带 shared expert column，并在 `_post_process_topk_ids` 中 remap 到 DeepEP interleaved layout。

Waterfill PR 最终结构:

```text
ServerArgs._handle_a2a_moe:
  if enable_deepep_waterfill and moe_a2a_backend != deepep:
    override moe_a2a_backend to deepep
  if DeepEP Waterfill enabled:
    force shared expert fusion
    support deepep_mode auto / normal / low_latency

TopK.__init__ with waterfill:
  if num_fused_shared_experts > 0:
    read global server_args.enable_deepep_waterfill
  if enabled:
    subtract the shared expert column from TopK config, so base TopK selects routed experts only
    clear num_fused_shared_experts in the base TopK config
    force STANDARD output
  keep deepep_waterfill_balancer = None

ModelRunner.initialize:
  load_model()
  _prepare_moe_topk()
    traverse TopK modules
    construct DeepEPWaterfillBalancer(num_routed_experts, world_size, rank, layer_id)
    assign module.deepep_waterfill_balancer
    call update_static_weights()

TopK.forward:
  base TopK:         [num_tokens, 8] routed ids after SGLang TopK postprocess
                    (logical if no EPLB remap, physical if dispatch info is active)
  Waterfill expand: [num_tokens, 9] SGLang/DeepEP physical/interleaved ids + shared expert slot
  DeepEP dispatch:  consumes expanded topk_ids/topk_weights
```

关键点:

- `TopK.__init__` 在 `num_fused_shared_experts > 0` 时读取 global server args。若 Waterfill 启用，内部把 `top_k` 减去 shared column，并把 `num_fused_shared_experts` 置 0，让 base TopK 只输出 routed expert columns；这些 ids 仍可能已被 EPLB postprocess remap。
- `ModelRunner._prepare_moe_topk()` 在模型加载后统一构造 `DeepEPWaterfillBalancer`，从 `ModelRunner` 读取 `moe_ep_size` / `moe_ep_rank`，从 model config 读取 `n_routed_experts`。
- `TopK.forward_*()` 调 base TopK，然后 `_apply_deepep_waterfill()` 调 `DeepEPWaterfillBalancer.expand_topk()` 扩成 9 列。
- `TopK.empty_topk_output()` 也统一走 `_apply_deepep_waterfill()`，模型侧不再手写 empty batch expansion。
- `DeepEPWaterfillBalancer` 根据 routed experts 的 rank load，为 shared expert 选择目标 rank。
- `rank_load` 目前主要用于 static EPLB 水位线模式；它只在 `init_expert_location` 提供 3D `logical_count` 时可用。低 batch static 路径会直接做 local shared expansion，跳过 routed-count kernel 和 all-reduce。
- dynamic 路径现在通过 SGLang 的 `moe_expert_parallel_all_reduce()` 聚合 routed counts 和本 rank token 数，保证 decode/extend 都参与同一 EP all-reduce。
- 算法文件内部会把 routed expert id remap 成 DeepEP interleaved layout，并写入 shared expert id。

### Important Gaps For Our POC

1. **PR TopK Waterfill path 仍没有 stage 输入**

   最终 `4e76f923a` 已把 Waterfill 从模型侧 wrapper 整理成 `server_args` 全局开关 + `ModelRunner._prepare_moe_topk()` + `TopK._apply_deepep_waterfill()`，但 `TopK.forward_*()` 仍只接收 `num_token_non_padded` 和 `expert_location_dispatch_info`，没有 `forward_batch.forward_mode`、stage 或 token mask。如果 policy config 限制只在 prefill 或只在 decode 启用，就不能直接复用 `--enable-deepep-waterfill` 作为最终 POC 接口；否则不匹配的 stage 仍可能进入 L2。这个改动说明 TopK remap 边界是好落点，但 POC 还需要显式 `moe_lb_context`。

   本文档里的 `SGLangRoutingContext` 就是这个 `moe_lb_context` 的 proposed adapter-only dataclass；它不是 SGLang 当前已有类型。

   不建议让 `TopK` 自己去全局读取 batch 信息。当前可见的 `get_forward_context()` 只在 piecewise CUDA graph runner 的 `set_forward_context(...)` 作用域内设置，不覆盖普通 eager/cuda graph forward，也不是 MoE TopK 的稳定 contract。`num_token_non_padded` 只能处理 padding，不能区分 `EXTEND`、`MIXED`、`DECODE`、`TARGET_VERIFY` 或 draft extend。`ExpertDistributionRecorder.with_forward_pass(...)` 虽然接收 `ForwardBatch`，但它是统计采集路径，没有提供 TopK 可依赖的 batch getter。

2. **`deepep_mode` 结论需要更新**

   之前文档里的 "DeepEP normal-only" 已经过期。最终 PR 文档、help 文案和 unit test 都确认 Waterfill 支持 `deepep_mode=auto`、`normal` 或 `low_latency`，并建议 production decode 用 `auto` 或 `low_latency` 保留 CUDA graph。最终文档还明确: 默认使用 static EPLB rank-load metadata，设置 `SGLANG_DISABLE_STATIC_WATERFILL=1` 才强制 dynamic all-reduce。POC 不应再默认拒绝 `low_latency`；只有具体 adapter path 尚未验证时才单独加 guard。

3. **`rank_load` metadata 已同步，PR 的指针式 static path 是合理的**

   最终 PR 已把 `rank_load` 计算放进 `ExpertLocationMetadata.init_by_eplb(...)`，但只接受 3D `logical_count`；mapping-only init 没有 static rank load，会 fallback 到 dynamic Waterfill。`ExpertLocationMetadata.update(...)` 会同步 `rank_load`。关键点是 PR 里的 balancer 持有的是 `rank_load` 指针/引用；只要 SGLang 更新的是同一个 buffer 的内容，`DeepEPWaterfillBalancer` 不需要每次重新 load 数据，也不会因为 prepare 阶段只设置一次而天然 stale。POC adapter 应保留这种 live-reference 语义。只有当某个 framework 替换了 rank-load 对象本身，而不是原地更新内容时，adapter 才需要在 rebalance 后刷新 `prior_rank_load` handle，或在 route 前 lazy-read 最新对象。

4. **算法文件直接依赖 SGLang 类型**

   最终 PR 把 `DeepEPWaterfillBalancer` 的构造移到 `ModelRunner._prepare_moe_topk()`，TopK 只保存并调用已注入的 balancer；Waterfill 激活则由 `TopK` 读取 global server args。依赖仍是 SGLang-specific: `model_runner.py` 读取 EP rank/world 和模型 config，`topk.py` 读取 global server args，`deepep_waterfill.py` 依赖 `envs`、`StandardTopKOutput` 和 `moe_expert_parallel_all_reduce()`，并访问 global expert location metadata / EP group。抽独立模块时，这些应移到 SGLang adapter。

   因此这里应该按两层理解 PR 的 `DeepEPWaterfillBalancer`:

   | 部分 | 归属 | 内容 |
   |---|---|---|
   | Waterfill core policy | core | 输入 current rank load / optional prior rank load / candidate ranks，输出 shared expert target rank |
   | DeepEP/SGLang wrapper | adapter | 负责 SGLang runtime prepare、static cache、EP collective、`StandardTopKOutput` rewrite、DeepEP interleaved id，以及把 core 输出的 shared route materialize 成 DeepEP expanded TopK 第 9 列 |

   这里要区分 semantic expansion 和 layout materialization。`top_k=8 -> routed top-k + shared route` 这个语义扩展可以属于 core；DeepEP dispatch 需要的 `[num_tokens, 9]` expanded TopK 是具体 layout。core 决定 shared route 应该发到哪个 EP rank、权重是多少；adapter 决定它在 `StandardTopKOutput.topk_ids/topk_weights` 里怎么编码。

   也就是说，Waterfill/Balance 的框架无关决策属于 core；PR 中和 SGLang 对象、DeepEP layout、通信绑定的包装部分属于 adapter。

## Recommended POC Interface

### ID-Space Contract

框架解耦的核心 contract 必须以 logical expert id 为输入:

```text
Core input:
  token-level logical_topk_ids/topk_weights + placement/rank snapshot + stage

Core output:
  selected replica / physical expert ids + possibly adjusted weights

SGLang adapter:
  logical ids <-> SGLang physical ids
  SGLang physical ids <-> DeepEP interleaved ids
  padding mask / num_token_non_padded
  materialize shared route into DeepEP expanded TopK
  EP process group and all-reduce
```

不能把 SGLang 普通 `TopK(..., expert_location_dispatch_info=...)` 之后的 `topk_ids` 当作 core logical input，因为 `_post_process_topk_ids` 已经可能完成了 `topk_ids_logical_to_physical(...)`。PR #19290 最终的 `--enable-deepep-waterfill` 能工作，是因为它是 SGLang/DeepEP-specific path，直接处理 SGLang 的 physical/interleaved layout；这不能泛化成框架解耦接口。

### L1 EPLB Adapter Interface

当前 POC 的 L1 只复用 SGLang EPLB，不实现新的 `PlacementPlanner`、per-layer replica budget 或 placement optimizer。L1 adapter 的职责是把 SGLang 现有 placement metadata 转成 core L2 可消费的 `PlacementSnapshot`。

```python
class SGLangPlacementAdapter:
    def snapshot(
        self,
        layer_id: int,
        metadata: ExpertLocationMetadata,
    ) -> PlacementSnapshot:
        ...
```

推荐插入点:

- 初始 placement: 读取 `compute_initial_expert_location_metadata(...)` 产出的 global `ExpertLocationMetadata`。
- runtime rebalance: 继续让 `EPLBManager.rebalance()` 调用 `ExpertLocationMetadata.init_by_eplb(...)` 和 `ModelRunner.update_expert_location(...)`。
- adapter 刷新: rebalance 后从 global metadata lazy read 或显式刷新 `PlacementSnapshot`。

`SGLangPlacementView` 是 adapter view，不是 core type。当前 L2 POC 只读取 SGLang EPLB metadata；core placement plan 是否导出成 `ExpertLocationMetadata` 暂时待定，等待 L1 placement 调研结果后再设计。

### L2 Prefill Interface

L2 core 应以 logical TopK 为输入，在 DeepEP dispatch 前由 adapter 改写 physical routing。`stage` 是 optional runtime context；某个 policy 是否只在 prefill、只在 decode、两者都运行或不按 stage gating，由 core 级 policy config 表达。`enabled_stages=None` 允许 `stage=None`，显式 stage set 则要求 stage 已知。当前某个 SGLang POC 可以选择只在纯 prefill/extend token slice 开启，但这不是 Waterfill 算法或 core API 的限制。

```python
@dataclass
class RoutingRequest:
    layer_id: int
    logical_topk_ids: Tensor
    topk_weights: Tensor
    placement: PlacementSnapshot
    stage: Optional[Literal["prefill", "decode", "mixed", "idle", "speculative"]]
    current_rank_load: Optional[Tensor]
    prior_rank_load: Optional[Tensor]
    token_count: Optional[Tensor]
    lb_token_mask: Optional[Tensor]
    routed_scaling_factor: float
    policy: RoutingPolicyConfig


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
    def route_tokens(
        self,
        request: RoutingRequest,
    ) -> RoutingDecision:
        ...
```

`logical_topk_ids` 不是 logical expert 的聚合分布。shared-expert Waterfill 最终需要的是 per-rank load，而不是必须需要 `physical_to_rank_map`。SGLang adapter 可以直接把 EPLB `rank_load` 映射成 `RoutingRequest.prior_rank_load`，也可以像 PR dynamic path 一样从 post-remap physical top-k 统计 routed counts 并 all-reduce 成 `RoutingRequest.current_rank_load`。只有当 core/adapter 要从 framework-neutral physical top-k 自己推导 rank load 时，才需要 `physical_to_rank_map`；只用 logical expert count 直接估算 rank load 会丢失 replicas 和 rank placement 信息。

PR dynamic path 之所以能不传 `physical_to_rank_map`，是因为它依赖 SGLang/DeepEP physical id 的分段编号规则: 同一个 rank 上的 physical experts 连续排列，所以 `target_rank = expert_id // experts_per_rank`。例如 `experts_per_rank = 4` 时，physical id `0..3` 属于 rank 0，`4..7` 属于 rank 1。这个规则属于 SGLang/DeepEP adapter 细节，不应变成 core 假设。

组合 `LPLB + Waterfill` 时，EP group 通信顺序要和 Waterfill-only dynamic 区分开:

```text
Waterfill-only static:
  EPLB rank_load -> prior_rank_load -> Waterfill

Waterfill-only dynamic:
  baseline routed physical top-k -> all-reduce routed counts -> current_rank_load -> Waterfill

LPLB + Waterfill:
  all-reduce global per-rank / per-expert load before LPLB
  -> LPLB chooses routed physical top-k
  -> derive post-LPLB current_rank_load
  -> Waterfill consumes post-LPLB current_rank_load
```

也就是说，组合策略里 EP group 通信应该发生在 LPLB 之前，并由 LPLB 和 Waterfill 共享一次通信结果；Waterfill 不应再单独做第二次 dynamic all-reduce。

这里的两个 rank-load 字段语义不同，且不要求同时提供:

| 字段 | 来源 | 用法 |
|---|---|---|
| `current_rank_load` | 当前 batch/window 的 routed top-k 统计，dynamic path 需要 EP all-reduce | dynamic Waterfill 的当前负载输入 |
| `prior_rank_load` | EPLB `rank_load` / 历史 logical_count 推导出的 per-rank 水位线 | static Waterfill 的负载估计 |

POC 默认应选择 static 或 dynamic 其中一种模式。

SGLang adapter 可以提供局部 helper，把 core decision 转回 SGLang 对象:

```python
@dataclass
class SGLangRoutingContext:
    policy: RoutingPolicyConfig
    stage: Optional[Literal["prefill", "decode", "mixed", "idle", "speculative"]]
    lb_token_mask: Optional[Tensor]
    num_token_non_padded: Optional[Tensor]


class SGLangRoutingAdapter:
    def route_topk(
        self,
        logical_topk_output: StandardTopKOutput,
        placement: SGLangPlacementView,
        context: SGLangRoutingContext,
    ) -> StandardTopKOutput:
        ...
```

推荐插入方式:

下面只描述 online forward data path 的调用顺序，不是完整模块归属清单。完整接入还包括 `server_args` 配置、`ModelRunner` prepare/inject、placement/rank-load refresh、EP communication 和 DeepEP layout writeback。

```text
Option A (preferred):
  DeepseekV2MoE.forward_deepep
    build minimal SGLangRoutingContext from forward_batch
    self.topk(..., moe_lb_context=context)

  TopK/_post_process_topk_ids
    logical_topk_ids are available before topk_ids_logical_to_physical(...)
    placement = placement_adapter.snapshot(
        layer_id,
        expert_location_dispatch_info or global ExpertLocationMetadata,
    )
    if context allows L2 for this pure prefill/extend batch:
        adapter calls core route_tokens(logical_topk_ids, placement, context)
    elif context is MIXED and adapter has an extend-token mask:
        route only the extend token slice and keep decode tokens unchanged
    else:
        bypass L2 and use baseline topk_ids_logical_to_physical(...)
    adapter applies padding mask + physical/interleaved/shared-slot conversion

Option B:
  DeepseekV2MoE.forward_deepep
    call TopK in logical-id mode or bypass default expert_location remap
    call sglang_adapter.route_topk(...) before self.experts(...)
    use only as a fallback because it is more model-path-specific and higher level
```

这个方案拆成两个动作:

| 位置 | 负责什么 | 不负责什么 |
|---|---|---|
| `forward_deepep` | 从 `ForwardBatch` 提取 optional stage / token mask，并携带 policy config，构造 `moe_lb_context` | 不做 TopK remap，不做 LPLB/Waterfill routing |
| `TopK` hook | 在 logical ids 还没被 remap 前，读取 placement metadata，并调用 adapter/core | 不自己猜 stage，不从全局状态隐式读取 `ForwardBatch` |

因此 “stage-aware TopK hook” 不是 SGLang 现有名词，而是指 TopK hook 额外拿到 `moe_lb_context` 后再决定是否 route。

这样比直接复用 PR 的 `--enable-deepep-waterfill` 路径更适合当前 POC，因为:

- stage guard 清楚，decode、mixed decode tokens 和 Metro 不会被误触发。
- `forward_batch.num_token_non_padded` 和 `forward_mode` 能显式进入 routing context。
- `SGLangRoutingContext` 只携带 per-forward 控制信息: optional stage、token mask、policy config。logical->physical mapping 这种 placement 大 tensor 由 adapter 在 TopK hook 处从 `ExpertLocationDispatchInfo` / `ExpertLocationMetadata` 读取，再转换成 `PlacementSnapshot` 传给 core。
- Waterfill、后续 LPLB 都可以共享同一个 `route_tokens()` 接口。
- EP all-reduce 是 SGLang runtime 细节，由 adapter 屏蔽；`LPLB + Waterfill` 组合时这次通信应在 LPLB 前发生一次，并把 post-LPLB rank load 传给 Waterfill。
- logical/physical id 边界清楚，核心算法不会误读 SGLang 已 remap 的 physical ids。
- 主要 routing 逻辑仍在 TopK remap 边界附近，不需要把 `forward_deepep` 变成完整 routing 编排层。

仍然需要小的 TopK/DeepSeek 接口改动:

- Waterfill 开启时，base TopK 必须输出 routed-only topk。PR 最终在 `TopK.__init__` 中通过 `num_fused_shared_experts > 0` + global `enable_deepep_waterfill` 把 shared column 从 base TopK config 中扣掉；POC 可以复用这个设计点，但应接到 generic adapter/policy，而不是 hardcode PR-specific balancer。
- POC 可以复用 PR 的 `ModelRunner` prepare/inject 模式初始化 SGLang adapter/router。注入到 `TopK` 的对象是 SGLang/DeepEP wrapper；这个 wrapper 内部调用 framework-neutral `TokenRouter` / `SharedExpertWaterfillPolicy` core，并负责把 core decision 写回 `StandardTopKOutput` / DeepEP layout。
- MoE expert layout 仍需保留 DeepEP shared fusion 的 expanded expert slots。
- `forward_deepep` 需要把 stage/token-mask context 传入 `self.topk(...)`，当前 PR 的 TopK forward 还没有这个参数。

PR 中不属于 generic TopK hook 本身的改动，也不是“什么都不放”。它们按三类落位:

- **Core**: 框架无关策略。
- **Adapter**: SGLang/DeepEP 数据结构、layout、通信和 writeback。
- **SGLang integration glue/config/runtime**: SGLang 侧必须加的接线代码，但不作为 core/adapter public API。

| PR 改动点 | POC 归属 | 处理方式 |
|---|---|---|
| Waterfill rank-selection policy | Core | 从 `DeepEPWaterfillBalancer` 中拆出 framework-independent policy |
| `ExpertLocationMetadata.rank_load` | Adapter | 映射成 `RoutingRequest.prior_rank_load`；优先保留 PR 的 live reference 语义，只有 rank-load 对象被替换时才刷新 handle |
| EP all-reduce | Adapter | Waterfill-only dynamic 可单独 all-reduce；`LPLB + Waterfill` 在 LPLB 前通信一次并共享 |
| DeepEP interleaved id / shared expert route column / empty output | Adapter | 把 core decision materialize 成 `StandardTopKOutput` |
| `ModelRunner._prepare_moe_topk()` | SGLang integration glue/runtime | 复用遍历和注入模式，注入 generic adapter/router |
| server args backend override / shared fusion guard | SGLang integration glue/config | 新 POC 开关复用这些合法性检查，不进入 core |
| `TopK.__init__` global flag / routed-only base TopK shape | SGLang integration glue / TopK wiring | 不把 global flag 作为 core interface；只保留 base TopK shape 约束并由 POC hook/context 控制 |

## Intrusion Map For Current POC

| Area | 必要改动 | 原因 |
|---|---|---|
| `server_args.py` | 新增 POC 开关，例如 `--enable-moe-load-balancer` / `--moe-lb-policy waterfill`；借鉴 PR 自动切到 DeepEP、强制 shared fusion；与 PR 对齐时允许 `deepep_mode=auto/normal/low_latency`，只有具体 adapter 不支持时再显式 guard | 控制入口和非法组合保护 |
| `model_runner.py` | 借鉴 PR 的 `_prepare_moe_topk()`，在 `load_model()` 后遍历 TopK modules 并注入 SGLang routing adapter/router；rebalance 后需要刷新 placement view，rank-load 则保留 live reference 或在对象替换时刷新 handle | runtime prepare、EP rank/world、模型遍历都属于 SGLang adapter 层 |
| `topk.py` / `_post_process_topk_ids` 或等价 TopK hook | 在 PR 新增的 TopK Waterfill apply 落点上扩展 `moe_lb_context`；在 logical ids 被 SGLang remap 成 physical ids 前调用 adapter，或允许 adapter 接管 remap | generic L2 必须看到 logical top-k，同时必须 stage-aware；当前 PR 的 Waterfill 是 stage-less global path |
| `deepseek_v2.py::forward_deepep` | 只构造并传递最小 `SGLangRoutingContext`，包含 optional stage / `num_token_non_padded` / optional token mask；fallback 方案才在这里直接调用 adapter | stage gating 由 core policy config 判断，避免把 forward path 变成 routing 编排层 |
| `expert_location.py` / adapter | 从 `ExpertLocationMetadata` 提供 per-layer `SGLangPlacementView`；把 placement mapping 转成 `PlacementSnapshot`，把 `rank_load` 映射成 `RoutingRequest.prior_rank_load` | L1/L2 边界需要拆清: placement 归 L1，rank load 归 L2 routing |
| `eplb_manager.py` | runtime rebalance 后通知/刷新 router placement view，或让 router lazy read global metadata；如果 rank-load 是原地更新的 live buffer，不需要重新 load；如果对象被替换，则同步刷新 handle | L2 需要看到 SGLang EPLB 最新 placement 和 rank-load |
| `layers/moe/deepep_waterfill.py` 或新模块 | 保留 Waterfill kernels，但剥离 SGLang global/env/TopK 类型依赖 | 后续抽独立模块 |

## Non-Goals In This File

- 不设计 decode/Metro。
- 不接入 L3 prediction/prefetch。
- 不处理 TensorRT-LLM/vLLM 生产级 adapter。

## Follow-Up In This File

- LPLB 具体改哪些 SGLang 文件需要补充；当前版本只保留 LPLB policy facade 和组合入口。
