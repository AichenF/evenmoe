import sys
import types

import pytest
import torch

from moe_load_balancer.core.types import WeightTransferRequest


class _FakeEventHandle:
    def __init__(self):
        self.waited = False

    def current_stream_wait(self):
        self.waited = True


class _FakeManager:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.constructed_pools = []
        self.update_placement_calls = []
        self.weight_sync_calls = []
        num_layers = kwargs["num_layers"]
        num_local_master = kwargs["num_local_master_experts"]
        num_local_redundant = kwargs["num_local_redundant_experts"]
        ep_size = kwargs["_test_ep_size"]
        self.physical_to_logical_map = torch.full(
            (num_layers, ep_size * (num_local_master + num_local_redundant)),
            -1,
            dtype=torch.int64,
        )
        self.local_replica_fc1_weight_buffer = None
        self.local_replica_fc2_weight_buffer = None

    def construct_local_master_ptr_pool(self, layer_id, w13_list, w2_list):
        self.constructed_pools.append((layer_id, list(w13_list), list(w2_list)))

    def update_placement_sparse(self, layer_id, topk_ids):
        self.update_placement_calls.append((layer_id, topk_ids))

    def weight_sync(self, layer_id, async_finish=True):
        self.weight_sync_calls.append(layer_id)
        return _FakeEventHandle()


class _FakeProcessGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@pytest.fixture
def fake_ultra_ep(monkeypatch):
    # _FakeManager needs to know ep_size to size physical_to_logical_map, but
    # the real Manager's constructor kwargs (mirrored exactly in
    # register_weights) carry no such key -- stash it via a thin wrapper
    # rather than changing the production call site to pass a test-only arg.
    def _manager_factory(**kwargs):
        return _FakeManager(**kwargs, _test_ep_size=_manager_factory.ep_size)

    fake_module = types.SimpleNamespace(Manager=_manager_factory)
    monkeypatch.setitem(sys.modules, "ultra_ep", fake_module)
    return _manager_factory


@pytest.fixture
def no_op_distributed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)


@pytest.mark.unit
def test_register_weights_constructs_manager_and_master_pools(fake_ultra_ep):
    from moe_load_balancer.policies.l3.ultraep import UltraEPWeightTransfer

    fake_ultra_ep.ep_size = 2
    num_layers, num_local_master, num_local_redundant = 2, 3, 1
    expert_weights = [
        [torch.randn(num_local_master + num_local_redundant, 4, 4) for _ in range(2)]
        for _ in range(num_layers)
    ]

    transfer = UltraEPWeightTransfer()
    transfer.register_weights(
        expert_weights,
        ep_size=2,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        process_group=_FakeProcessGroup(rank=0, size=2),
    )

    manager = transfer._manager
    assert manager.kwargs["num_local_master_experts"] == num_local_master
    assert manager.kwargs["num_local_redundant_experts"] == num_local_redundant
    assert manager.kwargs["num_layers"] == num_layers
    assert len(manager.constructed_pools) == num_layers
    layer_id, w13_list, _ = manager.constructed_pools[0]
    assert layer_id == 0
    assert len(w13_list) == num_local_master


@pytest.mark.unit
def test_register_weights_rejects_process_group_size_mismatch(fake_ultra_ep):
    from moe_load_balancer.policies.l3.ultraep import UltraEPWeightTransfer

    fake_ultra_ep.ep_size = 2
    expert_weights = [[torch.randn(4, 4, 4), torch.randn(4, 4, 4)]]
    transfer = UltraEPWeightTransfer()
    with pytest.raises(AssertionError):
        transfer.register_weights(
            expert_weights,
            ep_size=4,
            num_local_master_experts=3,
            num_local_redundant_experts=1,
            process_group=_FakeProcessGroup(rank=0, size=2),
        )


@pytest.mark.unit
def test_transfer_only_overwrites_assigned_redundant_slots(fake_ultra_ep, no_op_distributed):
    from moe_load_balancer.policies.l3.ultraep import UltraEPWeightTransfer

    num_local_master, num_local_redundant, ep_size = 2, 1, 2
    fake_ultra_ep.ep_size = ep_size
    num_local_physical = num_local_master + num_local_redundant
    expert_weights = [
        [
            torch.full((num_local_physical, 2, 2), 1.0),
            torch.full((num_local_physical, 2, 2), 2.0),
        ]
    ]

    transfer = UltraEPWeightTransfer()
    transfer.register_weights(
        expert_weights,
        ep_size=ep_size,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        process_group=_FakeProcessGroup(rank=0, size=ep_size),
    )
    manager = transfer._manager
    # This rank's one redundant slot declined -- the Manager leaves it -1
    # rather than "replicate onto yourself".
    manager.physical_to_logical_map[0, num_local_master] = -1
    manager.local_replica_fc1_weight_buffer = torch.full((1, 2, 2), 99.0)
    manager.local_replica_fc2_weight_buffer = torch.full((1, 2, 2), 98.0)

    original_w13_redundant = expert_weights[0][0][num_local_master:].clone()

    transfer.transfer(
        WeightTransferRequest(layer_id=0, routing_signal=torch.zeros(4, 1, dtype=torch.int64))
    )

    assert torch.equal(expert_weights[0][0][num_local_master:], original_w13_redundant)
    assert len(manager.update_placement_calls) == 1
    called_layer_id, called_topk_ids = manager.update_placement_calls[0]
    assert called_layer_id == 0
    assert torch.equal(called_topk_ids, torch.zeros(4, 1, dtype=torch.int64))
    assert manager.weight_sync_calls == [0]


@pytest.mark.unit
def test_transfer_copies_assigned_redundant_slot(fake_ultra_ep, no_op_distributed):
    from moe_load_balancer.policies.l3.ultraep import UltraEPWeightTransfer

    num_local_master, num_local_redundant, ep_size = 2, 1, 2
    fake_ultra_ep.ep_size = ep_size
    num_local_physical = num_local_master + num_local_redundant
    expert_weights = [
        [
            torch.full((num_local_physical, 2, 2), 1.0),
            torch.full((num_local_physical, 2, 2), 2.0),
        ]
    ]

    transfer = UltraEPWeightTransfer()
    transfer.register_weights(
        expert_weights,
        ep_size=ep_size,
        num_local_master_experts=num_local_master,
        num_local_redundant_experts=num_local_redundant,
        process_group=_FakeProcessGroup(rank=0, size=ep_size),
    )
    manager = transfer._manager
    manager.physical_to_logical_map[0, num_local_master] = 7
    manager.local_replica_fc1_weight_buffer = torch.full((1, 2, 2), 99.0)
    manager.local_replica_fc2_weight_buffer = torch.full((1, 2, 2), 98.0)

    transfer.transfer(
        WeightTransferRequest(layer_id=0, routing_signal=torch.zeros(4, 1, dtype=torch.int64))
    )

    assert torch.equal(expert_weights[0][0][num_local_master:], torch.full((1, 2, 2), 99.0))
    assert torch.equal(expert_weights[0][1][num_local_master:], torch.full((1, 2, 2), 98.0))
