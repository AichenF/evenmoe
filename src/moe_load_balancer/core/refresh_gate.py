"""Gates when a fast-refresh L1 policy's next placement solve should run.

UltraEP's paper design ties placement freshness to how much traffic has
actually been observed, not to wall-clock time -- see the reference SGLang
integration's ``moe_balance_refresh_interval`` (default 64 prefill batches).
This is that same batch-counted gate, framework-neutral: which forward passes
count as "representative" (big enough, the right execution stage, ...) is a
framework concern the caller decides before calling ``should_refresh``, not
something this class knows about.
"""

from __future__ import annotations

from typing import Dict


class RefreshGate:
    """Per-layer "N representative batches since the last refresh" gate."""

    def __init__(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError("RefreshGate requires a positive interval.")
        self._interval = interval
        self._batches_since_refresh: Dict[int, int] = {}

    def should_refresh(self, layer_id: int, *, representative: bool) -> bool:
        """Whether layer ``layer_id`` should be re-solved on this call.

        The first call for any ``layer_id`` always refreshes -- there is no
        placement for it yet. If that bootstrap call was not representative
        (too small a batch, warmup, ...), the next representative call
        refreshes again immediately rather than leaving a placement solved
        from thin data in place for a full interval. Otherwise a layer
        refreshes once at least ``interval`` representative batches have
        passed since its last one; non-representative batches do not count.
        """
        if layer_id not in self._batches_since_refresh:
            self._batches_since_refresh[layer_id] = 0 if representative else self._interval
            return True
        batches = min(self._batches_since_refresh[layer_id] + 1, self._interval)
        if batches >= self._interval and representative:
            self._batches_since_refresh[layer_id] = 0
            return True
        self._batches_since_refresh[layer_id] = batches
        return False
