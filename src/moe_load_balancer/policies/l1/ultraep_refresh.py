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


class RefreshGate:
    """Per-layer "N representative batches since the last refresh" gate.

    Split into two calls so a caller whose ``representative`` check is itself
    expensive (e.g. it needs a GPU-CPU sync to read a token count off a
    device tensor) only pays for it when a refresh could actually happen:
    call ``is_due(layer_id)`` every time (cheap, pure Python bookkeeping, no
    GPU access) and only compute ``representative`` + call ``should_refresh``
    when it returns True -- when it returns False, ``should_refresh`` would
    decline regardless of what ``representative`` turns out to be, so
    computing it is pure waste.
    """

    def __init__(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError("RefreshGate requires a positive interval.")
        self._interval = interval
        self._batches_since_refresh: dict[int, int] = {}

    def is_due(self, layer_id: int) -> bool:
        """Advance ``layer_id``'s call counter by one and report whether a
        refresh could happen this call.

        The counter lives here, not in ``should_refresh``, precisely so a
        caller can call this on every single pass without ever computing
        ``representative``: skipping this call on a "not yet due" pass would
        leave the counter permanently unadvanced, and the layer would never
        become due through natural progression.
        """
        if layer_id not in self._batches_since_refresh:
            return True
        batches = min(self._batches_since_refresh[layer_id] + 1, self._interval)
        self._batches_since_refresh[layer_id] = batches
        return batches >= self._interval

    def should_refresh(self, layer_id: int, *, representative: bool) -> bool:
        """Whether layer ``layer_id`` should be re-solved on this call.

        Call only immediately after ``is_due(layer_id)`` returned True for
        the same ``layer_id`` on this same pass -- this method trusts that
        the counter is already current and does not advance it itself.

        The first call for any ``layer_id`` always refreshes -- there is no
        placement for it yet. If that bootstrap call was not representative
        (too small a batch, warmup, ...), the layer is due again on its very
        next call rather than leaving a placement solved from thin data in
        place for a full interval. Otherwise a due call only actually
        refreshes when it is also representative; a due-but-not-
        representative call refreshes nothing and stays due next time.
        """
        if layer_id not in self._batches_since_refresh:
            self._batches_since_refresh[layer_id] = 0 if representative else self._interval
            return True
        if representative:
            self._batches_since_refresh[layer_id] = 0
            return True
        return False
