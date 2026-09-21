from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from moe_load_balancer import MoELoadBalancer, RoutingDecision, RoutingPipeline
from moe_load_balancer.adapters.sglang import (
    placement,
    runtime,
    to_placement_request,
    to_routing_request,
    to_sglang_routing_output,
)
from moe_load_balancer.adapters.sglang.runtime import _SGLangCollectives

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
def engine(monkeypatch):
    layout = {
        "ep_size": 2,
        "num_physical_experts": 4,
        "num_local_physical_experts": 2,
        "model_config_for_expert_location": SimpleNamespace(num_groups=1),
    }
    monkeypatch.setattr(placement, "_expert_layout", lambda context: layout)
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
    mlb = runtime.create_load_balancer(ctx)
    assert type(mlb) is MoELoadBalancer
    return mlb, ctx


def route(engine, ids=None):
    mlb, ctx = engine
    ids = torch.tensor([[0, 1]]) if ids is None else ids
    request = to_routing_request(
        context=ctx,
        layer_id=0,
        logical_topk_ids=ids,
        topk_weights=torch.ones_like(ids, dtype=torch.float32),
        token_count=len(ids),
        stage="prefill",
        routed_scaling_factor=1.0,
    )
    decision = mlb.route_tokens(request)
    return to_sglang_routing_output(decision, context=ctx)


def test_routing_reads_in_place_changes_and_replaced_metadata(engine):
    mlb, ctx = engine
    assert route(engine).topk_ids.tolist() == [[0, 1]]
    ctx.resources.expert_location_metadata.logical_to_rank_dispatch_physical_map.copy_(
        torch.tensor([[2, 3]])
    )
    assert route(engine).topk_ids.tolist() == [[2, 3]]
    ctx.resources.expert_location_metadata = metadata((0, 3))
    assert route(engine).topk_ids.tolist() == [[0, 3]]
    ctx.resources = SimpleNamespace(expert_location_metadata=metadata((2, 1)))
    assert route(engine).topk_ids.tolist() == [[2, 1]]


def test_empty_routing_still_reaches_pipeline(engine):
    mlb, ctx = engine
    mlb.route_tokens = Mock(wraps=mlb.route_tokens)
    output = route(engine, torch.empty((0, 2), dtype=torch.int64))
    assert output.topk_ids.shape == (0, 2)
    mlb.route_tokens.assert_called_once()
    ctx.resources.expert_distribution_recorder.on_select_experts.assert_not_called()


def test_missing_committed_metadata_fails_at_routing_not_construction(engine):
    mlb, ctx = engine
    ctx.resources.expert_location_metadata = None
    new_mlb = runtime.create_load_balancer(ctx)
    with pytest.raises(RuntimeError, match="committed expert metadata"):
        route((new_mlb, ctx))


def test_commit_refreshes_derived_state_only_on_explicit_notification(engine):
    mlb, ctx = engine
    mlb._routing_pipeline = RoutingPipeline.from_value("lplb")
    mlb.on_placement_committed = Mock()
    ctx.resources.expert_location_metadata = metadata((2, 3))
    mlb.on_placement_committed.assert_not_called()
    runtime.commit_placement(mlb, ctx, [0])
    snapshot = mlb.on_placement_committed.call_args.args[0]
    assert snapshot.default_physical_for_logical.tolist() == [2, 3]
    assert (
        snapshot.physical_to_logical_map.data_ptr()
        == ctx.resources.expert_location_metadata.physical_to_logical_map.data_ptr()
    )


def test_collective_transport_reads_current_group(engine):
    mlb, ctx = engine
    transport = _SGLangCollectives(ctx)
    replacement = Mock(rank_in_group=1, world_size=2)
    ctx.parallel.moe_ep_group = replacement
    payload = torch.ones(2)
    assert transport.rank == 1 and transport.world_size == 2
    assert transport.all_reduce_sum(payload) is replacement.all_reduce.return_value
    replacement.all_reduce.assert_called_once_with(payload)


def test_planning_reads_context_and_keeps_per_call_inputs_explicit(engine):
    mlb, ctx = engine
    counts = torch.tensor([[[10, 20]]])
    ctx.config_bag("exec").moe.eplb_algorithm = "deepseek_vec"
    active = torch.tensor([True, True])
    request = to_placement_request(counts, context=ctx, active_ranks=active, num_nodes=1)
    assert request.stats.logical_count is counts
    assert request.topology.active_ranks is active
    assert request.topology.num_nodes == 1
    assert request.policy.name == "deepseek_vec"
    assert request.topology.num_physical_experts == 4
    assert request.topology.num_local_physical_experts == 2
    ctx.parallel.nnodes = 3
    assert to_placement_request(counts, context=ctx).topology.num_nodes == 3


def test_two_contexts_do_not_share_placement_or_mutate_topk(engine):
    mlb, first = engine
    second = SimpleNamespace(resources=SimpleNamespace(expert_location_metadata=metadata((2, 3))))
    ids = torch.tensor([[0, 1]])
    assert route((mlb, first), ids).topk_ids.tolist() == [[0, 1]]
    assert route((mlb, second), ids).topk_ids.tolist() == [[2, 3]]
    assert route((mlb, first), ids).topk_ids.tolist() == [[0, 1]]
    assert ids.tolist() == [[0, 1]]


def test_shared_output_reads_current_context_geometry(engine):
    _, ctx = engine
    ids = torch.tensor([[0, 3]])
    decision = RoutingDecision(
        routed_physical_topk_ids=ids,
        topk_weights=torch.tensor([[0.7, 0.3]]),
        shared_expert_rank=torch.tensor([1]),
    )
    output = to_sglang_routing_output(decision, context=ctx)
    assert output.topk_ids.tolist() == [[0, 4, 5]]
    assert output.recorded_physical_topk_ids is ids

    ctx.resources.expert_location_metadata = SimpleNamespace(num_physical_experts=6, ep_size=2)
    output = to_sglang_routing_output(decision, context=ctx)
    assert output.topk_ids.tolist() == [[0, 4, 7]]
    assert output.recorded_physical_topk_ids is ids


def test_explicit_placement_values_override_context_defaults(engine):
    _, ctx = engine
    request = to_placement_request(
        torch.ones((1, 2)),
        context=ctx,
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=2,
        num_nodes=1,
        algorithm="deepseek_vec",
    )
    assert request.topology.num_physical_experts == 8
    assert request.topology.num_local_physical_experts == 4
    assert request.topology.num_groups == 2
    assert request.topology.num_nodes == 1
    assert request.policy.name == "deepseek_vec"
