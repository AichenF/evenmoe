"""Configuration for one composed L2 routing algorithm pipeline.

The serving framework supplies one algorithm expression.  MLB owns parsing,
validation, composition order, and the capabilities implied by that expression.
Framework adapters consume capabilities instead of branching on policy names.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from moe_load_balancer.core.types import RoutingPolicyConfig


_REPLICA_POLICIES = frozenset({"static", "dynamic", "fake", "lplb"})
_WATERFILL_POLICIES = frozenset({"waterfill", "waterfill_dynamic"})
_ALIASES = {"lp": "lplb"}


@dataclass(frozen=True)
class RoutingCapabilities:
    """Framework-visible requirements of a configured routing pipeline."""

    replica_policy: Optional[str] = None
    requires_post_topk_routing: bool = False
    requires_global_logical_count: bool = False
    requires_dynamic_rank_load: bool = False
    requires_placement_state: bool = False
    requires_rank_dispatch_map: bool = False
    routes_shared_expert: bool = False
    requires_deepep: bool = False

    @property
    def routes_replicas_inline(self) -> bool:
        return self.replica_policy is not None


@dataclass(frozen=True)
class RoutingPipeline:
    """Validated, canonical L2 routing configuration.

    Expressions use ``+`` for composition, for example ``lplb+waterfill``.
    Bare ``waterfill`` expands to ``static+waterfill`` so the framework only
    needs one selection parameter even when redundant experts are enabled.
    ``waterfill_dynamic`` selects the collective-backed Waterfill load view.
    """

    name: str
    policies: tuple[str, ...]
    post_topk_policies: tuple[RoutingPolicyConfig, ...]
    capabilities: RoutingCapabilities

    @classmethod
    def default_replica_routing(cls) -> RoutingPipeline:
        """Return the deterministic replica policy used with EPLB metadata."""

        return cls.from_value("static")

    @classmethod
    def from_value(cls, value: RoutingPipeline | str | None) -> RoutingPipeline:
        if isinstance(value, cls):
            return value
        if value is not None and not isinstance(value, str):
            raise TypeError(
                "RoutingPipeline.from_value expects RoutingPipeline, str, or None, "
                f"got {type(value).__name__}."
            )
        return cls._from_string(value)

    @classmethod
    @lru_cache(maxsize=None)
    def _from_string(cls, value: Optional[str]) -> RoutingPipeline:
        if value is None:
            return cls(
                name="disabled",
                policies=(),
                post_topk_policies=(),
                capabilities=RoutingCapabilities(),
            )
        if value.strip().lower() in {"", "disabled", "none"}:
            return cls(
                name="disabled",
                policies=(),
                post_topk_policies=(),
                capabilities=RoutingCapabilities(),
            )

        raw = tuple(part.strip().lower() for part in value.split("+"))
        if any(not part for part in raw):
            raise ValueError(f"Invalid routing algorithm expression {value!r}.")
        normalized = tuple(_ALIASES.get(part, part) for part in raw)
        supported = _REPLICA_POLICIES | _WATERFILL_POLICIES
        unknown = sorted(set(normalized) - supported)
        if unknown:
            raise ValueError(
                f"Unknown routing algorithm(s) {unknown}; supported algorithms are "
                f"{sorted(supported)}. Use '+' to compose algorithms."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Routing algorithm expression contains duplicates: {value!r}.")

        replica = tuple(policy for policy in normalized if policy in _REPLICA_POLICIES)
        waterfill = tuple(policy for policy in normalized if policy in _WATERFILL_POLICIES)
        if len(replica) > 1:
            raise ValueError(
                "A routing pipeline can contain at most one physical-replica algorithm."
            )
        if len(waterfill) > 1:
            raise ValueError("A routing pipeline can contain at most one Waterfill mode.")

        # Waterfill consumes physical IDs.  Static replica selection is the
        # deterministic default when the user selects Waterfill by itself.
        replica_policy = replica[0] if replica else ("static" if waterfill else None)
        waterfill_policy = waterfill[0] if waterfill else None
        if replica_policy == "lplb" and waterfill_policy == "waterfill_dynamic":
            raise ValueError(
                "waterfill_dynamic cannot follow lplb: LPLB already produces the "
                "global routed-load view consumed by static Waterfill."
            )

        policies = tuple(
            policy for policy in (replica_policy, waterfill_policy) if policy is not None
        )
        canonical = "+".join(policies)
        inline_replica = replica_policy if replica_policy in {"static", "dynamic", "fake"} else None
        post_topk = []
        if replica_policy == "lplb":
            post_topk.append(RoutingPolicyConfig(name="lplb"))
        if waterfill_policy is not None:
            post_topk.append(RoutingPolicyConfig(name="waterfill"))

        capabilities = RoutingCapabilities(
            replica_policy=inline_replica,
            requires_post_topk_routing=bool(post_topk),
            requires_global_logical_count=replica_policy == "lplb",
            requires_dynamic_rank_load=waterfill_policy == "waterfill_dynamic",
            requires_placement_state=replica_policy == "lplb",
            requires_rank_dispatch_map=inline_replica == "static",
            routes_shared_expert=waterfill_policy is not None,
            requires_deepep=waterfill_policy is not None,
        )
        return cls(
            name=canonical,
            policies=policies,
            post_topk_policies=tuple(post_topk),
            capabilities=capabilities,
        )
