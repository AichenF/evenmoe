from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from moe_load_balancer.core.types import RoutingStage, TensorLike


@dataclass(frozen=True)
class SGLangRoutingContext:
    """Minimal per-forward context passed from SGLang model code to TopK.

    ``stage`` is normalized to :class:`RoutingStage` at the adapter boundary —
    upstream callers may pass strings, but should funnel them through
    :meth:`RoutingStage.from_str` before constructing this dataclass.
    """

    stage: RoutingStage
    policy: str
    num_token_non_padded: Optional[TensorLike] = None
    lb_token_mask: Optional[TensorLike] = None
