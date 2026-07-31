# MoE Load Balancer L1/L2 重构设计决议

状态：已确认，作为后续 MLB 重构的设计依据

确认日期：2026-07-31

实施范围：仅修改 `moe_load_balancer` 仓库；本轮不修改 SGLang fork

## 1. 唯一待定项

SGLang 后续需要一个集中承载 MLB 接入信息的变量域或对象。

本轮只确认“存在这样一个集中承载位置”，不讨论也不决定：

- 类型和名称；
- 字段组织；
- 创建和销毁时机；
- 所有权和生命周期；
- 它与 SGLang 其他 runtime 对象的关系。

除此之外，本文中的架构边界和重构方向均已确认。

## 2. 总体架构

### 2.1 每个模型或 engine 使用一个通用 MoELoadBalancer

SGLang 创建并持有一个通用的 `MoELoadBalancer` 实例：

```python
mlb = MoELoadBalancer(...)
```

同一个实例同时提供 L1 和 L2 能力：

```python
mlb.plan_placement(...)  # L1
mlb.route_tokens(...)    # L2
```

不再为每个框架和算法组合创建单独的 runtime，例如：

- `SGLangMoELoadBalancer`
- `SGLangEPLBRuntime`
- `LPLBRuntime`
- `WaterfillRuntime`
- SGLang 与某个具体算法绑定的其他 runtime

这样可以保持实现成本为“M 个框架无关 Policy + N 个框架 Adapter”，而不是 M×N 个框架和算法的组合实现。

### 2.2 MLB 不选择框架 Adapter

`MoELoadBalancer` 和框架无关 Policy 不接受 `framework="sglang"` 一类参数，也不 import、注册或查找 SGLang Adapter。

调用方向固定为：

```text
SGLang Glue
    ↓
前置 SGLang Adapter
    ↓
MoELoadBalancer
    ↓
后置 SGLang Adapter
    ↓
SGLang Glue
```

由 SGLang Glue 显式组织调用：

```python
request = sglang_adapter.to_placement_request(sglang_input)
plan = mlb.plan_placement(request)
sglang_maps = sglang_adapter.to_sglang_maps(plan)
```

Adapter 是无状态的数据翻译函数，不持有 `MoELoadBalancer`，也不控制工作流。

## 3. L1 EPLB 的职责边界

### 3.1 SGLang 负责框架生命周期和真实状态

以下能力留在 SGLang：

- TopK、dispatcher 等位置的 token 计数埋点；
- 一次完整 forward 的开始和结束；
- recorder 的临时计数、窗口、dump 和 reset；
- every-N cadence；
- 各 rank 统计结果的 all-reduce；
- physical count 到 logical utilization 的汇总；
- threshold 判断和是否跳过本次 rebalance；
- 初始专家权重加载；
- 运行时专家权重迁移；
- layer chunk 和 missing weight 恢复；
- 维护真正生效的 `ExpertLocationMetadata`；
- 决定新 placement 何时正式生效。

Threshold 明确保留在 SGLang：

```text
未达到阈值
→ SGLang 跳过本次 rebalance
→ 不调用 MLB EPLB Policy
```

### 3.2 L1 前置 Adapter 只翻译输入

只有当 SGLang 已经完成 dump、跨 rank 聚合和 threshold 判断，并决定执行 rebalance 时，才调用前置 Adapter：

```text
SGLang 已完成的统计结果
        ↓
PlacementStats
        ↓
PlacementRequest
```

前置 Adapter 不负责：

- recorder dump；
- all-reduce；
- threshold；
- cadence；
- 保存历史 observation；
- 加载或迁移权重。

### 3.3 EPLB Policy 只计算候选布局

统一入口为：

```python
plan = mlb.plan_placement(request)
```

`MoELoadBalancer` 根据请求选择框架无关的 placement policy。当前已有的主要实现是 `DeepSeekEPLBPlanner`。

Policy 输出 `PlacementPlan`。它只表示候选布局，不表示布局已经生效。

`PlacementPlan` 直接保存候选 mapping，例如：

- `physical_to_logical_map`
- `logical_to_all_physical_map`
- `logical_to_physical_count`
- 必要的算法 metadata

`PlacementPlan` 不再包含 `PlacementSnapshot`。

### 3.4 L1 后置 Adapter 只翻译输出

后置 Adapter 将候选 `PlacementPlan` 翻译为 SGLang 加载或迁移权重所需的 mapping：

```text
PlacementPlan
      ↓
SGLang mapping
```

它不调用 SGLang updater，也不移动权重。

### 3.5 只有真实权重更新后 placement 才生效

SGLang 接收候选 mapping 后：

1. 创建或更新 placement metadata；
2. 加载或迁移真实专家权重；
3. 处理 missing weight；
4. 全部成功后更新生效的 `ExpertLocationMetadata`；
5. 后续 forward 自然读取新的 metadata。

因此：

```text
L1 计算出 PlacementPlan
≠
新 placement 已经生效
```

候选 plan 不能在权重迁移完成前发布给 L2。L2 必须继续使用旧的真实 metadata，直到 SGLang 完成权重更新。

### 3.6 冷启动和周期性 rebalance 使用同一条算法路径

两种场景统一经过：

```text
SGLang Glue
→ 前置 Adapter
→ MoELoadBalancer.plan_placement()
→ EPLB Policy
→ 后置 Adapter
→ SGLang 加载或迁移权重
```

两者只在 SGLang 的触发时机上不同，不需要两套 EPLB runtime。

## 4. PlacementPlan 与 PlacementSnapshot

### 4.1 两者语义

- `PlacementPlan`：L1 计算得到、尚未应用的候选布局。
- `PlacementSnapshot`：SGLang 中当前真实生效、可供 L2 使用的布局。

MLB 不保存第二份可变的 SGLang placement 状态。

### 4.2 PlacementSnapshot 由 live metadata 无状态生成

不再维护有状态的 `SGLangPlacementAdapter` 或 `PlacementProvider`。改为无状态转换函数：

```python
snapshot = sglang_adapter.to_placement_snapshot(
    live_metadata,
    layer_id,
)
```

Snapshot 表达：

- layer id；
- logical/physical expert 数量；
- EP size；
- 每个 rank 的 physical expert 数量；
- physical-to-logical mapping；
- logical-to-physical candidates；
- 每个 logical expert 的副本数量。

能够引用 SGLang tensor view 时不复制数据。

对于当前连续布局，可以通过 physical expert id 和 `num_local_physical_experts` 推导 rank，不需要在每次 forward 构造并缓存 `physical_to_rank_map`。以后支持非连续布局时，再允许传入显式 rank map。

## 5. L2 LPLB + Waterfill 统一方案

### 5.1 统一接入位置

LPLB 和 Waterfill 进入同一次 L2 调用：

```text
SGLang Gate
→ logical TopK
→ 统一调用 MLB L2
→ routed physical experts + shared expert rank
→ DeepEP dispatch
```

精确位置是在 `select_experts` 的 logical expert 选择完成后、logical-to-physical 和 DeepEP 最终 ID materialization 之前。

### 5.2 固定 Policy 顺序

```text
SGLang logical TopK
    ↓
统一 L2 前置 Adapter
    ├─ logical TopK → RoutingRequest
    ├─ live metadata → PlacementSnapshot
    └─ SGLang 准备 LPLB 所需的全局 logical count
    ↓
MoELoadBalancer.route_tokens()
    ↓
可选 LPLB
logical expert IDs → routed physical expert IDs
    ↓
通用 post-LPLB 统计
physical expert IDs → per-rank routed load
    ↓
可选 Waterfill
根据最新 per-rank routed load 选择 shared expert rank
    ↓
RoutingDecision
    ├─ routed physical TopK IDs
    ├─ routed weights
    └─ shared expert rank
    ↓
统一 L2 后置 Adapter / SGLang Glue
    ├─ recorder 记录 routed physical IDs
    ├─ append shared expert
    ├─ DeepEP interleaved ID 转换
    └─ 构造 StandardTopKOutput
    ↓
DeepEP dispatch
```

执行顺序固定为：

```text
LPLB
→ 根据 LPLB 输出重新统计 rank load
→ Waterfill
```

Waterfill 不重复 Gate 或 TopK，也不读取 placement metadata。它只使用 LPLB 路由完成后的 per-rank routed load。

### 5.3 通信

组合模式采用：

1. SGLang 为 LPLB 对 logical expert count 做一次 EP all-reduce；
2. LPLB 得到新的 physical routing；
3. 根据 physical routing 在本地统计每个 rank 的 routed load；
4. Waterfill 使用该负载，不执行第二次 all-reduce。

独立 dynamic Waterfill 可以保留自己的统计路径；LPLB + Waterfill 组合路径不重复通信。

### 5.4 Recorder 与 shared expert

Recorder 只记录：

```python
decision.routed_physical_topk_ids
```

后续 append 的 shared expert 不进入 EPLB placement 统计。

### 5.5 空 token rank

即使某个 rank 当前没有 token，它仍然必须进入统一 L2 hook，并参与 LPLB 所需的 collective，保证各 rank collective 顺序一致。

## 6. 需要删除的旧设计

重构中删除：

- `SGLangEPLBRuntime`
- `set_default_runtime`
- `get_default_runtime`
- `ensure_default_runtime`
- `clear_default_runtime`
- MLB 内的 `EPLBRecorder`
- `SGLangStatsBridge`
- `SGLangStatsCollector`
- `StatsCollector` protocol
- `RebalancePolicyConfig`
- `PlacementStats.average_utilization_rate`
- `EPLBPlacementProvider`
- `PlacementProvider` protocol
- 有状态的 `SGLangPlacementAdapter`
- `PlacementPlan.snapshots`
- `provider.update_with(plan)` 一类提前发布候选 placement 的行为
- 算法与框架绑定的 runtime 和 registry
- 上述旧 API 的 exports、测试和文档

不保留旧 API 兼容层。

## 7. MLB 重构实施顺序

### 阶段一：整理核心数据模型

- 重写 `PlacementPlan`，只保存候选 mapping；
- 明确 `PlacementSnapshot` 只代表已生效布局；
- 删除 threshold、recorder、provider 等框架生命周期类型；
- 更新包级 exports；
- 清理旧 API 测试。

### 阶段二：加入统一 Orchestrator

实现：

```python
class MoELoadBalancer:
    def plan_placement(self, request) -> PlacementPlan:
        ...

    def route_tokens(self, request) -> RoutingDecision:
        ...
```

约束：

- 不 import SGLang；
- 不根据 framework 字段选择 Adapter；
- L1 根据 policy 选择 placement planner；
- L2 按顺序执行可组合的 policy pipeline；
- 同一个实例服务 L1 和 L2。

### 阶段三：重写无状态 SGLang Adapter

L1：

```python
to_placement_request(...)
to_sglang_maps(...)
```

L2 placement：

```python
to_placement_snapshot(...)
```

L2 routing：

```python
to_routing_request(...)
to_sglang_routing_output(...)
```

Adapter 只转换数据，不保存 runtime 或 metadata，不管理 recorder、threshold、collective、权重迁移或 SGLang 生命周期。

### 阶段四：调整 Policy

- 保留并重构 `DeepSeekEPLBPlanner`，使其返回新的 `PlacementPlan`；
- 保证 DeepSeek EPLB 算法结果不变；
- 将 LPLB 和 Waterfill 作为框架无关的 L2 Policy；
- 加入 LPLB 后统一统计 physical-to-rank load 的通用步骤；
- 固定组合顺序为 LPLB → rank-load count → Waterfill；
- 本轮不补齐尚未实现的其他 L1 算法。

### 阶段五：清理旧实现

物理删除不再使用的 runtime、registry、recorder、bridge 和 provider 文件，并清理对应 imports、exports、types、测试和冲突文档。

### 阶段六：验证

L1 至少覆盖：

- 前置 Adapter 正确生成 request；
- Orchestrator 正确选择 EPLB Policy；
- 后置 Adapter 正确生成 SGLang mapping；
- 冷启动和周期 rebalance 使用同一算法入口；
- 候选 plan 不会提前成为 L2 snapshot；
- 只有 live SGLang metadata 改变后 snapshot 才改变；
- DeepSeek EPLB 重构前后结果一致。

L2 至少覆盖：

- LPLB → count → Waterfill 的执行顺序；
- Waterfill 接收 LPLB 调整后的 rank load；
- 只启用 LPLB、只启用 Waterfill、同时启用和同时关闭；
- recorder 输入不包含 shared expert；
- 空 token rank 仍走 collective-compatible 路径；
- live metadata 正确转换为 `PlacementSnapshot`。

最后运行相关 CPU 单元测试、格式与静态检查以及 `git diff --check`。完整测试若因环境缺少 Triton 无法收集，应明确记录环境限制；真正接入 SGLang 后还需要 GPU 和多 rank 集成测试。

## 8. 本轮不做

- 不修改 SGLang fork；
- 不设计 SGLang 集中变量域或对象；
- 不决定该对象的名称、字段、类型和生命周期；
- 不补齐其他尚未实现的 L1 算法；
- 不保留旧 API 兼容层；
- 不在 MLB 实现 SGLang recorder、threshold 或权重迁移；
- 不在本轮合并 SGLang 中现有的 LPLB/Waterfill 调用点。

MLB 本轮先完成 L1 目标边界、统一 Orchestrator、无状态 SGLang Adapter，以及 LPLB + Waterfill 的统一 L2 Policy Pipeline；SGLang 侧接线在后续单独完成。
