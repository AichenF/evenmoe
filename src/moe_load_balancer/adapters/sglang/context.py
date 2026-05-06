from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from moe_load_balancer.core.types import RoutingPolicyConfig, RoutingStage, TensorLike


@dataclass(frozen=True)
class SGLangRoutingContext:
    """Minimal per-forward context passed from SGLang model code to TopK.

    ``stage`` is optional runtime context. Upstream callers that have stage
    information should normalize it with :meth:`RoutingStage.from_str` before
    constructing this dataclass; callers without reliable stage information can
    leave it as ``None``.
    """

    policy: RoutingPolicyConfig | str
    stage: Optional[RoutingStage] = None
    num_token_non_padded: Optional[TensorLike] = None
    lb_token_mask: Optional[TensorLike] = None

    def __post_init__(self) -> None:
        # Keep adapter construction compatible with older string policy call
        # sites. RoutingRequest repeats this coercion so adapter callers cannot
        # accidentally bypass core-side validation.
        object.__setattr__(self, "policy", RoutingPolicyConfig.from_value(self.policy))
