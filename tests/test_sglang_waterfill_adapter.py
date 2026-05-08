import pytest

from moe_load_balancer.adapters.sglang import (
    SGLangWaterfillDispatchPlan,
    build_dynamic_waterfill_dispatch_plan,
    build_static_waterfill_dispatch_plan,
)


class _TensorLikeRankLoad:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        raise AssertionError("tolist() should not be used in tensor-like dispatch plan")

    def __add__(self, other):
        other_values = other.values
        return _TensorLikeRankLoad([left + right for left, right in zip(self.values, other_values)])

    def __len__(self):
        return len(self.values)

    def sum(self):
        return sum(self.values)

    def max(self):
        return max(self.values)


@pytest.mark.unit
class TestSGLangWaterfillAdapter:
    def test_static_dispatch_plan_keeps_rank_load_reference(self):
        rank_load = [10, 1]

        plan = build_static_waterfill_dispatch_plan(rank_load)

        assert isinstance(plan, SGLangWaterfillDispatchPlan)
        assert plan.rank_load is rank_load
        assert plan.allow_all_ranks
        assert plan.target_total == 0

    def test_dynamic_dispatch_plan_matches_sglang_effective_load(self):
        plan = build_dynamic_waterfill_dispatch_plan(
            [8, 8],
            [2, 2],
            topk_size=8,
        )

        assert plan.rank_load == [10, 10]
        assert plan.allow_all_ranks
        assert plan.target_total == 11

    def test_dynamic_dispatch_plan_can_limit_to_candidate_ranks(self):
        plan = build_dynamic_waterfill_dispatch_plan(
            [100, 8],
            [0, 0],
            topk_size=8,
        )

        assert plan.rank_load == [100, 8]
        assert not plan.allow_all_ranks
        assert plan.target_total == 61

    def test_dynamic_dispatch_plan_preserves_tensor_like_effective_rank_load(self):
        routed_load = _TensorLikeRankLoad([8, 8])
        token_count = _TensorLikeRankLoad([2, 2])

        plan = build_dynamic_waterfill_dispatch_plan(
            routed_load,
            token_count,
            topk_size=8,
        )

        assert isinstance(plan.rank_load, _TensorLikeRankLoad)
        assert plan.rank_load.values == [10, 10]
        assert plan.allow_all_ranks
        assert plan.target_total == 11
