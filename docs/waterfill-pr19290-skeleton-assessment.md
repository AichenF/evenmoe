# Waterfill PR #19290 Skeleton Assessment

本文档记录 PR #19290 的 shared-expert Waterfill 如何放进当前
`moe_load_balancer` skeleton。已经通过 skeleton 调整解决的问题不再放进
“未解决项”，只保留当前还需要实现或验证的部分。

## Current Baseline

PR #19290 的 Waterfill 思路可以放进当前 skeleton，但不能把 SGLang 里的
`DeepEPWaterfillBalancer` 原样搬进 `policies/l2/`。当前已定稿的拆分是:

| 层 | 归属 | 当前职责 |
|---|---|---|
| Core contract | `core/` | `RoutingRequest`、`RoutingDecision`、`RoutingPolicyConfig`、optional `RoutingStage` |
| L2 load state | `core/l2.py` + `core/types.py` | `RoutingLoadProvider` / `RoutingLoadView` 暴露 prior rank-load 等长持 L2 state |
| L2 policy | `policies/l2/` | Waterfill 根据 current/prior rank load 和 candidate ranks 选择 shared expert target rank |
| SGLang adapter | `adapters/sglang/` | SGLang/DeepEP 类型、id layout、TopK writeback、EP all-reduce、no-copy rank-load view |
| SGLang integration glue | SGLang patch | server args、backend guard、`ModelRunner._prepare_moe_topk()`、TopK hook 接线 |

当前 skeleton 已经覆盖的设计点:

- `RoutingRequest.stage` 是 optional。
- `RoutingPolicyConfig.enabled_stages` 在 core 里表达通用 stage gating。
- Waterfill 可以 stage-agnostic 使用，也可以配置成只在 prefill/decode 等 stage 运行。
- `current_rank_load` 表达当前 batch/window 的即时 per-rank load。
- `prior_rank_load` 不再作为每次 request 的输入，而是通过 `RoutingLoadProvider` /
  `RoutingLoadView` 作为 L2 长持 state 提供。
- 对 PR 这种原地更新的 `ExpertLocationMetadata.rank_load`，`RoutingLoadView` 必须
  no-copy 暴露 live reference，不能复制到新 storage。

## Remaining Open Items

下面是把 PR #19290 真正放进 skeleton 时仍然需要解决的部分。

### 1. Implement Framework-Neutral Waterfill Policy

**Status: unresolved**

当前 skeleton 只有接口，还没有 `policies/l2/waterfill.py`。

需要实现的 core policy 只应该做 rank-selection:

```text
input:
  logical_topk_ids
  request.current_rank_load, or load_provider.load_view(layer_id).prior_rank_load
  optional token mask
  source rank / candidate rank view

output:
  shared_expert_rank[token] = target_ep_rank
```

不能放进 policy 的内容:

- `StandardTopKOutput`。
- DeepEP interleaved expert id。
- SGLang process group / all-reduce。
- `TopK` module mutation。
- server args / backend guard。

### 2. Build The Real SGLang TopK Hook

**Status: unresolved**

当前 `SGLangRoutingAdapter` 仍是 skeleton stub。真正接入 SGLang 时需要在 logical id
还没被 `_post_process_topk_ids` remap 成 physical id 之前调用 adapter，或者让 adapter
显式接管 logical-to-physical remap。

需要确认的实现点:

- hook 位置能拿到 logical top-k ids 和 weights。
- hook 能拿到 layer id。
- hook 能拿到 `ExpertLocationDispatchInfo` 或 global `ExpertLocationMetadata`。
- hook 能在 baseline remap 前进入，否则 core 会错把 SGLang physical ids 当成 logical ids。

### 3. Materialize Shared Expert Route Into SGLang/DeepEP Layout

**Status: unresolved**

Core 只应该输出:

```text
shared_expert_rank[token] = rank_id
```

SGLang adapter 还需要把这个 rank 写回 DeepEP 所需的 expanded TopK layout:

```text
shared_expert_id = target_rank * new_experts_per_rank + old_experts_per_rank
```

剩余工作:

- 把 `[num_tokens, top_k]` 扩成 DeepEP shared-fusion 需要的 `[num_tokens, top_k + 1]`。
- 写 `StandardTopKOutput.topk_ids/topk_weights`。
- 处理 empty output path。
- 保持 logits / routed expert top-k parity。

### 4. Keep ID Namespaces Strictly Separated

**Status: unresolved**

PR 路径里至少有三种 id:

- logical expert id。
- SGLang/EPLB plain physical expert id。
- DeepEP interleaved physical expert id。

当前 skeleton contract 已经定义了边界，但 adapter 还没有实现转换。实际实现时必须保证:

- core input 是 logical expert id。
- `RoutingDecision.routed_physical_topk_ids` 使用 `PlacementSnapshot` 定义的 plain physical id namespace。
- DeepEP interleaved id 只能在 SGLang adapter materialization 阶段生成。

### 5. Implement Dynamic Rank-Load Communication In Adapter

**Status: unresolved**

Dynamic Waterfill 需要当前 batch/window 的 global per-rank load。这个通信不能进入 core。

目标调用链:

```text
adapter local count
  -> adapter EP all-reduce
  -> RoutingRequest.current_rank_load
  -> core Waterfill policy
```

组合 LPLB + Waterfill 时，需要共享一次通信结果:

```text
all-reduce global per-rank / per-expert load before LPLB
  -> LPLB chooses routed physical top-k
  -> derive post-LPLB current_rank_load
  -> Waterfill consumes post-LPLB current_rank_load
```

不能让 LPLB 和 Waterfill 各自做一次 EP all-reduce。

### 6. Implement No-Copy SGLang RoutingLoadProvider

**Status: partially resolved in contract, unresolved in SGLang implementation**

Skeleton 已经规定 `RoutingLoadView.prior_rank_load` 是 no-copy live reference。但真实
`SGLangRoutingLoadProvider.load_view(layer_id)` 还没有实现。

实现要求:

- 对 PR #19290 的 static path，必须暴露原始 `ExpertLocationMetadata.rank_load` storage。
- 可以做字段映射 / wrapper。
- 如果 slicing 某层，必须确认 slicing 仍是 view，不是 copy。
- 不能 `.clone()`、`.copy()`、`.cpu()` 后交给 L2。
- 如果某个 framework 替换 rank-load 对象本身，而不是原地更新内容，provider 才需要刷新 handle。

### 7. Add End-To-End Integration Tests

**Status: unresolved**

当前已有 API contract tests，但还没有真实 SGLang/DeepEP behavior tests。

至少需要覆盖:

- disabled policy 与 baseline TopK parity。
- Waterfill stage-agnostic path。
- stage-gated Waterfill path。
- static Waterfill 使用 live `prior_rank_load`，并且原地更新后 policy 读到新值。
- dynamic Waterfill 使用 `current_rank_load`。
- id namespace conversion。
- shared expert 第 9 列 materialization。
- empty output path。
- LPLB + Waterfill 只共享一次 communication result。

## Not Open Anymore

以下问题已经通过 skeleton 调整解决，不再作为 open item:

- stage 是否必须区分 prefill/decode: 已改成 optional stage + core-level `enabled_stages`。
- Waterfill 是否必须绑定某个 stage: 已支持 stage-agnostic policy。
- `stage` 和 `enabled_stages` 是否重复: 已明确 `stage` 是 request context，`enabled_stages` 是 policy gating config。
- `prior_rank_load` 是否应该每次塞进 `RoutingRequest`: 已改成 L2 长持 `RoutingLoadProvider`。
- PR 的 static rank-load cache 是否必然 stale: 已确认 PR 使用 live pointer，skeleton 也要求 no-copy view。

## Final Assessment

当前 skeleton 的方向仍然合理。剩余问题主要不是 core contract 设计，而是实际接入层:

- Waterfill policy 还没有实现。
- SGLang TopK hook 还没有实现。
- SGLang/DeepEP layout materialization 还没有实现。
- dynamic communication 和 LPLB+Waterfill 共享通信还没有实现。
- no-copy `RoutingLoadProvider` 还没有对真实 SGLang metadata 落地。

PR #19290 的核心 Waterfill 思路可以作为 L2 policy 进入模块；PR 中和 SGLang/DeepEP
对象、通信、layout、runtime injection 绑定的部分应继续留在 SGLang adapter /
integration glue。
