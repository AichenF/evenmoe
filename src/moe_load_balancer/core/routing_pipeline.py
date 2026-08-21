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


_INPUT_TRANSFORMS = frozenset({"fake"})
_REPLICA_POLICIES = frozenset({"static", "dynamic", "dynamic_random", "lplb"})
_WATERFILL_POLICIES = frozenset({"waterfill", "waterfill_dynamic"})
_ALIASES = {"lp": "lplb"}


def _resolves_dispatch(replica_policy):
    """Whether the policy realises its own per-token choice.

    Resolving is not automatically the cheaper side of the boundary. A runtime
    that fuses selection with expert-load recording -- vLLM does -- can apply a
    share table inside the one kernel it was already going to launch, so a
    policy expressible as shares costs nothing extra there. Handing back
    per-token ids means the policy runs its own kernel first and the framework
    still runs its recording pass: two launches where there was one.

    SGLang has no such fusion -- its native replica step is one gather outside
    any kernel -- which is why the same policy is free there and is not here.

    ``MLB_L2_DISPATCH=shares`` forces the share-table side for the policies
    that can express themselves that way, so the two boundaries can be measured
    against each other instead of argued about.
    """
    import os

    if replica_policy is None:
        return False
    if os.environ.get("MLB_L2_DISPATCH", "").strip().lower() == "shares":
        if replica_policy in ("lplb", "static"):
            return False
    return replica_policy in ("lplb", "static", "dynamic", "dynamic_random")



@dataclass(frozen=True)
class DeploymentTopology:
    """What a framework can tell MLB about a deployment before it starts.

    Deliberately small: this is consulted while the framework is still reading
    its configuration, so it may only contain what is known that early.
    """

    num_redundant_experts: int = 0
    routes_shared_expert: bool = False
    """Whether the framework can dispatch the shared expert to a chosen rank.

    False on runtimes that replicate it per rank, where there is no such
    decision to hand back.
    """


@dataclass(frozen=True)
class RoutingCapabilities:
    """Framework-visible requirements of a configured routing pipeline.

    Every field exists so the framework can act on a *declaration* instead of
    on an assumption about what a policy is. A framework that keys behaviour off
    a policy name both taxes policies that do not need the behaviour and leaves
    policies that do need it unable to ask.
    """

    requires_post_topk_routing: bool = False
    requires_placement_state: bool = False
    requires_rank_dispatch_map: bool = False
    routes_shared_expert: bool = False
    requires_deepep: bool = False

    # -- inputs the pipeline consumes -------------------------------------
    resolves_dispatch: bool = False
    """Whether the policy turns its own decision into per-token ids.

    False means the decision is a per-expert distribution the framework can
    apply itself, which a runtime that fuses selection with load recording does
    in one pass. True means the policy realises it -- the only option for an
    algorithm whose choice depends on the token and not just on the expert, and
    the reason the selection step has to be separable at all.

    Both remain available: a framework keeps its built-in selection for when no
    policy is configured, and a policy that declares this replaces only the
    selection, never the recording.
    """

    consumes_global_logical_count: bool = False
    """Whether any policy reads the EP-wide per-logical token count.

    Distinct from ``requires_post_topk_routing``: a replica policy can run
    without ever looking at the load, and gathering the count for it costs a
    collective per layer per forward that is then discarded.
    """

    max_input_staleness_steps: int = 0
    """How old the consumed inputs may be, in steps. 0 means current.

    A framework that can only supply older data must say so rather than
    silently substituting it; a policy that cannot tolerate the age must be
    able to refuse.
    """

    # -- runtime properties the framework has to respect -------------------
    supports_concurrent_microbatches: bool = True
    """False when policies keep per-layer state that concurrent micro-batches
    would clobber."""

    graph_stability: str = "stable"
    """``stable``, or ``realloc_on_placement_change`` when a placement update
    can replace policy tensors and invalidate a captured graph."""

    placement_state_lifecycle: str = "none"
    """``none``, ``per_placement`` or ``per_forward`` -- how long
    placement-derived policy state stays valid."""


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
    input_transform: Optional[str]
    capabilities: RoutingCapabilities

    def is_applicable(self, topology: DeploymentTopology) -> Optional[str]:
        """Why this pipeline cannot act on a deployment, or None if it can.

        Returns a reason rather than a bool so a framework that disables the
        pipeline can tell the operator *what* it disabled and why. A silent
        bool would turn a diagnosable configuration into a mystery.

        Applicability is per policy, not per layer: LPLB needs replicated
        experts to have a split to decide, Waterfill needs the framework to
        dispatch the shared expert at all, and neither condition follows from
        the other. Deciding both from one framework-side assumption is how a
        policy that had work to do gets switched off.
        """

        if "lplb" in self.policies and topology.num_redundant_experts == 0:
            return (
                "lplb divides a logical expert's traffic among its replicas, "
                "and this deployment has no redundant experts"
            )
        if self.capabilities.routes_shared_expert and not topology.routes_shared_expert:
            return (
                "waterfill assigns the shared expert to a rank, and this "
                "runtime does not dispatch the shared expert"
            )
        return None

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
                input_transform=None,
                capabilities=RoutingCapabilities(),
            )
        if value.strip().lower() in {"", "disabled", "none"}:
            return cls(
                name="disabled",
                policies=(),
                post_topk_policies=(),
                input_transform=None,
                capabilities=RoutingCapabilities(),
            )

        raw = tuple(part.strip().lower() for part in value.split("+"))
        if any(not part for part in raw):
            raise ValueError(f"Invalid routing algorithm expression {value!r}.")
        normalized = tuple(_ALIASES.get(part, part) for part in raw)
        supported = _INPUT_TRANSFORMS | _REPLICA_POLICIES | _WATERFILL_POLICIES
        unknown = sorted(set(normalized) - supported)
        if unknown:
            raise ValueError(
                f"Unknown routing algorithm(s) {unknown}; supported algorithms are "
                f"{sorted(supported)}. Use '+' to compose algorithms."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Routing algorithm expression contains duplicates: {value!r}.")

        transforms = tuple(policy for policy in normalized if policy in _INPUT_TRANSFORMS)
        replica = tuple(policy for policy in normalized if policy in _REPLICA_POLICIES)
        waterfill = tuple(policy for policy in normalized if policy in _WATERFILL_POLICIES)
        if len(transforms) > 1:
            raise ValueError("A routing pipeline can contain at most one input transform.")
        if len(replica) > 1:
            raise ValueError(
                "A routing pipeline can contain at most one physical-replica algorithm."
            )
        if len(waterfill) > 1:
            raise ValueError("A routing pipeline can contain at most one Waterfill mode.")

        input_transform = transforms[0] if transforms else None
        explicit_replica = replica[0] if replica else None
        # Fake is an experimental pre-TopK workload transform.  Its historical
        # physical routing behavior is Dynamic when no explicit replica router
        # is present.  Bare Waterfill keeps Static as its deterministic default.
        replica_policy = explicit_replica
        if replica_policy is None and input_transform == "fake":
            replica_policy = "dynamic"
        if replica_policy is None and waterfill:
            replica_policy = "static"
        waterfill_policy = waterfill[0] if waterfill else None
        if replica_policy == "lplb" and waterfill_policy == "waterfill_dynamic":
            raise ValueError(
                "waterfill_dynamic cannot follow lplb: LPLB already produces the "
                "global routed-load view consumed by static Waterfill."
            )

        execution_policies = tuple(
            policy for policy in (replica_policy, waterfill_policy) if policy is not None
        )
        canonical_parts = []
        if input_transform is not None:
            canonical_parts.append(input_transform)
        if explicit_replica is not None:
            canonical_parts.append(explicit_replica)
        elif input_transform is None and replica_policy is not None:
            canonical_parts.append(replica_policy)
        if waterfill_policy is not None:
            canonical_parts.append(waterfill_policy)
        canonical = "+".join(canonical_parts)

        post_topk = []
        if replica_policy is not None:
            post_topk.append(RoutingPolicyConfig(name=replica_policy))
        if waterfill_policy is not None:
            post_topk.append(
                RoutingPolicyConfig(
                    name="waterfill",
                    metadata={
                        "load_view": (
                            "global" if waterfill_policy == "waterfill_dynamic" else "local"
                        )
                    },
                )
            )

        # Only LPLB solves against the EP-wide load; the other replica policies
        # decide from placement alone. Declaring that here is what lets a
        # framework skip the collective rather than gather it for every
        # post-TopK policy.
        consumes_count = replica_policy == "lplb" or waterfill_policy is not None
        capabilities = RoutingCapabilities(
            requires_post_topk_routing=bool(post_topk),
            requires_placement_state=replica_policy == "lplb",
            requires_rank_dispatch_map=replica_policy == "static",
            routes_shared_expert=waterfill_policy is not None,
            requires_deepep=waterfill_policy is not None,
            # LPLB owns a dispatch kernel that realises its LP solution, and
            # static and dynamic each pick one replica per token. Letting them
            # resolve is what keeps the selection step something an algorithm
            # can own rather than something the framework's kernel decides.
            resolves_dispatch=_resolves_dispatch(replica_policy),
            consumes_global_logical_count=consumes_count,
            # LPLB's LP is a per-batch decision, but the distribution it tracks
            # moves slowly; one step of age is acceptable and is what lets the
            # collective producing it live outside a CUDA graph.
            max_input_staleness_steps=1 if consumes_count else 0,
            supports_concurrent_microbatches=replica_policy != "lplb",
            graph_stability=(
                "realloc_on_placement_change" if replica_policy == "lplb" else "stable"
            ),
            placement_state_lifecycle=(
                "per_placement" if replica_policy == "lplb" else "none"
            ),
        )
        return cls(
            name=canonical,
            policies=execution_policies,
            post_topk_policies=tuple(post_topk),
            input_transform=input_transform,
            capabilities=capabilities,
        )
