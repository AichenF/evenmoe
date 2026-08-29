import pytest

from moe_load_balancer import RefreshGate


@pytest.mark.unit
def test_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="positive"):
        RefreshGate(0)


@pytest.mark.unit
def test_first_call_always_refreshes():
    gate = RefreshGate(interval=4)
    assert gate.should_refresh(0, representative=False) is True


@pytest.mark.unit
def test_waits_the_full_interval_of_representative_batches():
    gate = RefreshGate(interval=3)
    gate.should_refresh(0, representative=True)  # bootstrap, resets counter to 0

    assert gate.should_refresh(0, representative=True) is False  # 1
    assert gate.should_refresh(0, representative=True) is False  # 2
    assert gate.should_refresh(0, representative=True) is True  # 3 -> refresh


@pytest.mark.unit
def test_non_representative_batches_do_not_count():
    gate = RefreshGate(interval=2)
    gate.should_refresh(0, representative=True)  # bootstrap

    assert gate.should_refresh(0, representative=False) is False
    assert gate.should_refresh(0, representative=False) is False
    assert gate.should_refresh(0, representative=False) is False
    assert gate.should_refresh(0, representative=True) is True


@pytest.mark.unit
def test_a_non_representative_bootstrap_refreshes_again_on_the_next_real_batch():
    """A placement solved from a tiny/unrepresentative first batch is likely
    poor; the next representative batch should fix it immediately rather than
    waiting out a full interval with a known-thin placement in place."""
    gate = RefreshGate(interval=5)
    gate.should_refresh(0, representative=False)  # bootstrap, thin data

    assert gate.should_refresh(0, representative=False) is False
    assert gate.should_refresh(0, representative=True) is True


@pytest.mark.unit
def test_layers_are_tracked_independently():
    gate = RefreshGate(interval=2)
    gate.should_refresh(0, representative=True)  # layer 0 bootstrap
    assert gate.should_refresh(1, representative=True) is True  # layer 1 bootstrap

    assert gate.should_refresh(0, representative=True) is False
    assert gate.should_refresh(1, representative=True) is False
    assert gate.should_refresh(0, representative=True) is True
    assert gate.should_refresh(1, representative=True) is True
