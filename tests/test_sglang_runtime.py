from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from moe_load_balancer import MoELoadBalancer, RoutingPipeline
from moe_load_balancer.adapters.sglang.runtime import _SGLangCollectives, _SGLangRuntime

pytestmark = pytest.mark.unit


def metadata(mapping=(0, 1)):
    return SimpleNamespace(
        ep_size=2,
        num_logical_experts=2,
        num_physical_experts=4,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]]]),
        logical_to_all_physical_map_num_valid=torch.tensor([[2, 2]]),
        logical_to_rank_dispatch_physical_map=torch.tensor([mapping]),
    )


@pytest.fixture
def mlb(monkeypatch):
    layout = {
        "ep_size": 2,
        "num_physical_experts": 4,
        "num_local_physical_experts": 2,
        "model_config_for_expert_location": SimpleNamespace(num_groups=1),
    }
    monkeypatch.setattr(_SGLangRuntime, "_expert_layout", lambda self: layout)
    config = SimpleNamespace(
        moe=SimpleNamespace(moe_load_balancer_algorithm="static", eplb_algorithm="deepseek")
    )
    ctx = SimpleNamespace(
        config_bag=lambda name: config,
        parallel=SimpleNamespace(moe_ep_rank=0, nnodes=2, moe_ep_group=Mock()),
        resources=SimpleNamespace(
            expert_location_metadata=metadata(),
            expert_distribution_recorder=Mock(),
            experts_capturer=Mock(),
        ),
    )
    mlb = MoELoadBalancer.from_sglang_context(ctx)
    assert type(mlb) is MoELoadBalancer
    assert mlb.runtime_context is ctx
    return mlb


def route(mlb, ids=None):
    ids = torch.tensor([[0, 1]]) if ids is None else ids
    return mlb.route_topk(
        layer_id=0,
        logical_topk_ids=ids,
        topk_weights=torch.ones_like(ids, dtype=torch.float32),
        router_logits=None,
        token_count=len(ids),
        stage="prefill",
        routed_scaling_factor=1.0,
    )


def test_routing_reads_in_place_changes_and_replaced_metadata(mlb):
    ctx = mlb.runtime_context
    assert route(mlb).topk_ids.tolist() == [[0, 1]]
    ctx.resources.expert_location_metadata.logical_to_rank_dispatch_physical_map.copy_(
        torch.tensor([[2, 3]])
    )
    assert route(mlb).topk_ids.tolist() == [[2, 3]]
    ctx.resources.expert_location_metadata = metadata((0, 3))
    assert route(mlb).topk_ids.tolist() == [[0, 3]]
    replacement_recorder = Mock()
    ctx.resources.expert_distribution_recorder = replacement_recorder
    output = route(mlb)
    assert (
        replacement_recorder.on_select_experts.call_args.kwargs["topk_ids"]
        is output.recorded_physical_topk_ids
    )
    assert (
        ctx.resources.experts_capturer.capture.call_args.kwargs["topk_indices"]
        is output.recorded_physical_topk_ids
    )


def test_empty_routing_still_reaches_pipeline_and_recorder(mlb):
    mlb.route_tokens = Mock(wraps=mlb.route_tokens)
    output = route(mlb, torch.empty((0, 2), dtype=torch.int64))
    assert output.topk_ids.shape == (0, 2)
    mlb.route_tokens.assert_called_once()
    mlb.runtime_context.resources.expert_distribution_recorder.on_select_experts.assert_called_once()


def test_missing_committed_metadata_fails_at_routing_not_construction(mlb):
    mlb.runtime_context.resources.expert_location_metadata = None
    new_mlb = MoELoadBalancer.from_sglang_context(mlb.runtime_context)
    with pytest.raises(RuntimeError, match="committed expert metadata"):
        route(new_mlb)


def test_commit_refreshes_derived_state_only_on_explicit_notification(mlb):
    mlb._routing_pipeline = RoutingPipeline.from_value("lplb")
    mlb.on_placement_committed = Mock()
    mlb.runtime_context.resources.expert_location_metadata = metadata((2, 3))
    mlb.on_placement_committed.assert_not_called()
    mlb.commit_placement([0])
    snapshot = mlb.on_placement_committed.call_args.args[0]
    assert snapshot.default_physical_for_logical.tolist() == [2, 3]
    assert (
        snapshot.physical_to_logical_map.data_ptr()
        == mlb.runtime_context.resources.expert_location_metadata.physical_to_logical_map.data_ptr()
    )


def test_collective_transport_reads_current_group(mlb):
    transport = _SGLangCollectives(mlb.runtime_context)
    replacement = Mock(rank_in_group=1, world_size=2)
    mlb.runtime_context.parallel.moe_ep_group = replacement
    payload = torch.ones(2)
    assert transport.rank == 1 and transport.world_size == 2
    assert transport.all_reduce_sum(payload) is replacement.all_reduce.return_value
    replacement.all_reduce.assert_called_once_with(payload)


def test_planning_reads_context_and_keeps_per_call_inputs_explicit(mlb):
    mlb.plan_placement = Mock(wraps=mlb.plan_placement)
    counts = torch.tensor([[[10, 20]]])
    mlb.runtime_context.config_bag("exec").moe.eplb_algorithm = "deepseek_vec"
    active = torch.tensor([True, True])
    mlb.plan_placement_from_counts(counts, active_ranks=active, use_flat_topology=True)
    request = mlb.plan_placement.call_args.args[0]
    assert request.stats.logical_count is counts
    assert request.topology.active_ranks is active
    assert request.topology.num_nodes == 1
    assert request.policy.name == "deepseek_vec"
