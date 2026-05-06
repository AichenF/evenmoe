# Waterfill PR #19290 Skeleton Assessment

本文档整理 PR #19290 的 shared-expert Waterfill 如何放进当前 `moe_load_balancer`
skeleton，以及直接搬入时会遇到的不适配点。

## Short Answer

PR #19290 可以放进当前 skeleton，但不能把 SGLang 里的
`DeepEPWaterfillBalancer` 原样搬进 `policies/l2/`。这个 PR 当前把三类职责揉在一起:

- Waterfill rank-selection policy。
- SGLang TopK / `StandardTopKOutput` writeback。
- DeepEP id layout、EP all-reduce、server args、ModelRunner 注入。

进入 skeleton 时应该拆成:

| 层 | 归属 | 内容 |
|---|---|---|
| Core contract | `core/` | `RoutingRequest`、`RoutingDecision`、`RoutingPolicyConfig`、optional `RoutingStage` |
| L2 policy | `policies/l2/` | Waterfill 根据 rank load / candidate ranks 选择 shared expert target rank |
| SGLang adapter | `adapters/sglang/` | `StandardTopKOutput` 转换、DeepEP interleaved id、shared expert 第 9 列、EP all-reduce |
| SGLang integration glue | SGLang patch | server args、backend guard、`ModelRunner._prepare_moe_topk()`、TopK hook 接线 |

## What Fits Well

当前 skeleton 的几个核心设计和 PR #19290 是匹配的:

- `RoutingRequest.logical_topk_ids` 可以承载 remap 前的 logical TopK。
- `RoutingRequest.current_rank_load` 可以承载 dynamic Waterfill 的当前 batch 全局 rank load。
- `RoutingLoadView.prior_rank_load` 可以承载 EPLB / historical rank-load prior，并作为 L2 长持 state 供 policy 读取。
- `RoutingDecision.shared_expert_rank` 可以表达 core 的 Waterfill 决策: shared expert 应该 dispatch 到哪个 EP rank。
- `PlacementSnapshot.logical_to_physical_candidates` / `physical_to_rank_map` 可以支持 LPLB 或 future routed-replica policy。
- `SGLangRoutingAdapter` 的方向正确: build `RoutingRequest` -> call core router -> materialize `RoutingDecision` back into SGLang layout。

## Required Split

### 1. Core

Core 只定义稳定、框架中立的 API。

应该留在 core:

- `RoutingStage`: optional runtime context。
- `RoutingPolicyConfig`: policy 名称和通用 stage gating。
- `RoutingRequest`: logical TopK、placement、rank-load 输入。
- `RoutingLoadView` / `RoutingLoadProvider`: prior rank-load 等 L2 长持 load state。
- `RoutingDecision`: framework-neutral routing decision。

不应该进入 core:

- `StandardTopKOutput`。
- `ExpertLocationMetadata`。
- DeepEP interleaved id。
- SGLang process group / all-reduce。
- `--enable-deepep-waterfill`。

### 2. Policy

Waterfill policy 应该只实现框架无关的决策:

```text
input:
  logical_topk_ids
  request-local current_rank_load or load-source prior_rank_load
  optional token mask
  source rank / candidate rank view

output:
  shared_expert_rank
```

Policy 可以选择使用或忽略 `request.stage`。Waterfill 本身不应该被强制绑定到
prefill 或 decode。是否只在某些 stage 启用，由 core 级
`RoutingPolicyConfig.enabled_stages` 表达。

### 3. SGLang Adapter

SGLang adapter 负责所有 SGLang/DeepEP 细节:

- 在 SGLang logical id 被 `_post_process_topk_ids` remap 成 physical id 之前捕获 logical TopK，或显式接管 remap。
- 从 `ExpertLocationMetadata` / `ExpertLocationDispatchInfo` 构造 `PlacementSnapshot`。
- 如果走 dynamic Waterfill，在 adapter 内做 EP all-reduce，填充 `RoutingRequest.current_rank_load`。
- 如果走 static Waterfill，从 EPLB metadata 读取 rank load，暴露为 `RoutingLoadView.prior_rank_load`。
- 把 `RoutingDecision.shared_expert_rank` materialize 成 DeepEP shared expert id。
- 把 routed ids remap 成 DeepEP interleaved layout。
- 追加 DeepEP dispatch 需要的第 9 列。

### 4. SGLang Integration Glue

PR 里的这些改动属于 SGLang integration，而不是 skeleton core/policy:

- `server_args.py` 的 `enable_deepep_waterfill`。
- 自动 override `moe_a2a_backend=deepep`。
- 强制 shared expert fusion。
- `ModelRunner._prepare_moe_topk()` 遍历 TopK 并注入 runtime object。
- `TopK.__init__` 里把 base TopK 调整成 routed-only columns。

这些可以借鉴，但不要变成 core public API。

## Main Incompatibilities

### 1. PR 是 stage-less global feature

PR 当前由 `--enable-deepep-waterfill` 全局启用，`TopK.forward_*()` 没有拿到
`forward_batch.forward_mode`。这意味着它无法天然区分 prefill、decode、mixed、
idle 或 speculative。

Skeleton 应该这样处理:

- `RoutingRequest.stage: Optional[RoutingStage] = None`。
- `RoutingPolicyConfig.enabled_stages` 是 core 级 gating。
- `enabled_stages=None` 表示 policy 不按 stage 限制。
- 显式列出所有 stage 和 `enabled_stages=None` 不完全等价: 前者仍要求
  `request.stage` 已知，后者允许 `stage=None`。
- `enabled_stages={PREFILL}` 表示只在 prefill 运行。
- `enabled_stages={DECODE}` 表示只在 decode 运行。
- `enabled_stages={PREFILL, DECODE}` 表示两者都运行。

Adapter 只负责把框架的 runtime stage 翻译成 `RoutingStage`。算法是否允许在该
stage 运行，应由 core 里的 policy config 判断。

### 2. PR 混合了 core policy 和 SGLang layout

`DeepEPWaterfillBalancer.expand_topk()` 同时做:

- rank load counting。
- EP all-reduce。
- target rank 选择。
- DeepEP id remap。
- `StandardTopKOutput` writeback。

拆进 skeleton 时，只有 “target rank 选择” 属于 policy。其他都属于 adapter 或
integration glue。

### 3. ID namespace 容易混淆

PR 中至少有三种 id namespace:

- logical expert id。
- SGLang/EPLB plain physical expert id。
- DeepEP interleaved physical expert id。

Skeleton 必须明确:

- core input 是 logical id。
- `RoutingDecision.routed_physical_topk_ids` 使用 `PlacementSnapshot` 定义的
  plain physical id namespace。
- DeepEP interleaved id 只能由 SGLang adapter 生成。

### 4. Shared expert column 不能进入 core layout

PR 里 DeepEP Waterfill 把 shared expert 作为第 9 个 routed slot dispatch。
这个 `[N, 8] -> [N, 9]` 的 tensor layout 是 DeepEP/SGLang 细节。

Core 只应该表达:

```text
shared_expert_rank[token] = rank_id
```

Adapter 再决定:

```text
shared_expert_id = target_rank * new_experts_per_rank + old_experts_per_rank
```

以及把这一列写到 `StandardTopKOutput.topk_ids/topk_weights`。

### 5. all-reduce 责任必须留在 adapter

Core 不应该 import 或调用 SGLang communication APIs。

如果 Waterfill 需要当前 batch 的 global rank load:

```text
adapter local count
  -> adapter EP all-reduce
  -> RoutingRequest.current_rank_load
  -> core Waterfill policy
```

组合 LPLB + Waterfill 时，也应该共享一次通信结果，避免 LPLB 和 Waterfill 各做一次
all-reduce。

### 6. Static rank load cache 的真实边界

PR 在 `ExpertLocationMetadata` 上新增 `rank_load`，并在 `update(...)` 中同步。
这里需要注意: PR 的 static Waterfill path 传入的是 `rank_load` 的指针/引用，
不是每次复制出来的一份旧数组。只要 SGLang 在 rebalance/update 时原地更新这块
rank-load 内容，`DeepEPWaterfillBalancer` 后续读到的就是更新后的值。因此在 PR
当前实现里，把 `rank_load` 保存到 balancer 里是合理的，不能把它简单归类成
"prepare 一次后必然 stale" 的问题。

Skeleton 版仍需要把语义写清楚:

- `RoutingLoadView.prior_rank_load` 可以是 live tensor/reference；core policy 通过 provider/source 读取字段内容，
  不要求 adapter 每次 deep copy。
- 如果某个 framework 像 PR 一样原地更新 rank-load buffer，adapter 可以长期持有这个
  reference，不需要每次重新 load。
- 如果某个 framework 不是原地更新，而是替换成新的 tensor/list/object，adapter 才需要
  在 rebalance 后刷新 runtime handle，或者每次 route 前 lazy-read 最新对象。
- 测试应确认 `prior_rank_load` 真的影响 Waterfill 决策，同时确认 live reference 语义
  不会被 skeleton 误改成 deep-copy snapshot。

## Skeleton Adjustments

已经确认应该采用的 contract:

```python
@dataclass(frozen=True)
class RoutingRequest:
    layer_id: int
    logical_topk_ids: TensorLike
    topk_weights: TensorLike
    placement: PlacementSnapshot
    stage: Optional[RoutingStage] = None
    current_rank_load: Optional[TensorLike] = None
    token_count: Optional[TensorLike] = None
    lb_token_mask: Optional[TensorLike] = None
    routed_scaling_factor: float = 1.0
    policy: RoutingPolicyConfig = field(default_factory=RoutingPolicyConfig)


@dataclass(frozen=True)
class RoutingLoadView:
    prior_rank_load: Optional[TensorLike] = None
```

```python
@dataclass(frozen=True)
class RoutingPolicyConfig:
    DISABLED: ClassVar[str] = "disabled"

    name: str = DISABLED
    enabled_stages: Optional[frozenset[RoutingStage]] = None

    @property
    def is_disabled(self) -> bool:
        return self.name == "disabled"

    def allows_stage(self, stage: Optional[RoutingStage]) -> bool:
        if self.enabled_stages is None:
            return True
        if stage is None:
            return False
        return stage in self.enabled_stages

    def should_run_for_stage(self, stage: Optional[RoutingStage]) -> bool:
        return (not self.is_disabled) and self.allows_stage(stage)
```

This supports:

- Waterfill on all stages: `enabled_stages=None`。
- Waterfill only prefill: `enabled_stages={RoutingStage.PREFILL}`。
- Waterfill only decode: `enabled_stages={RoutingStage.DECODE}`。
- Waterfill prefill + decode: `enabled_stages={RoutingStage.PREFILL, RoutingStage.DECODE}`。

## Recommended Implementation Sequence

1. Keep core contracts framework-neutral.
2. Add `RoutingPolicyConfig` and optional `RoutingRequest.stage`.
3. Implement `policies/l2/waterfill.py` as a pure rank-selection policy.
4. Implement SGLang placement adapter: EPLB metadata -> `PlacementSnapshot` + prior rank load.
5. Implement SGLang routing adapter: logical TopK -> `RoutingRequest` -> `RoutingDecision` -> `StandardTopKOutput`.
6. Add SGLang TopK hook before logical-to-physical remap, or make adapter own that remap.
7. Add SGLang stage context plumbing, but allow `stage=None`.
8. Add tests for:
   - disabled policy parity;
   - `enabled_stages` gating;
   - Waterfill with `stage=None`;
   - Waterfill only prefill;
   - Waterfill only decode;
   - id namespace conversion;
   - `prior_rank_load` actually changing target rank selection.

## Final Assessment

当前 skeleton 的方向是合理的:

- `core/` 定义通用 contract。
- `policies/` 放 Waterfill/LPLB/Metro 等算法。
- `adapters/sglang/` 处理 SGLang/DeepEP 对象和 layout。

PR #19290 证明这个拆分是必要的。它的核心 Waterfill 思路可以作为 L2 policy 进入
模块，但 PR 当前实现形态是 SGLang-specific runtime feature，不能直接作为通用
module API。
