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
_REPLICA_POLICIES = frozenset({"static", "dynamic", "dynamic_random", "lplb", "ultraep"})
_WATERFILL_POLICIES = frozenset({"waterfill", "waterfill_dynamic"})
_ALIASES = {"lp": "lplb"}


@dataclass(frozen=True)
class ExpertDeploymentConfig:
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

    carries_placement_metadata: bool = False
    """Whether the framework threads ``PlacementPlan.metadata`` into the
    committed ``PlacementSnapshot.metadata``.

    An L1 policy may publish state its own L2 half needs -- ``ultraep`` puts its
    per-replica token quota there. Nothing forces a framework to carry it: the
    plan and the snapshot are separate objects the framework builds at different
    times, and a runtime whose committed placement has no room for extra fields
    has to thread it explicitly. MLB's vLLM adapter takes the quota as an
    argument to ``to_placement_snapshot``; its SGLang adapter has no such channel
    today, so ``ultraep`` is not usable there yet.

    Declared as a capability rather than inferred from the runtime's name so a
    framework that gains the plumbing only has to flip this, and so a policy that
    needs the channel can ask without knowing which runtime it is under.

    Defaults False: a framework that has not considered the question cannot be
    assumed to carry the metadata, and silently dropping it degrades the policy
    rather than failing it.
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
    dispatch_fixed_by_placement: bool = False
    """Whether the per-token answer is a pure function of the committed placement.

    True means the policy's choice does not depend on the batch at all, so a
    framework can materialise it once when a placement commits and never call
    the pipeline per forward. ``static`` is the case: it resolves a replica per
    logical expert from the placement, and re-deriving that on every forward
    buys nothing.

    This exists because crossing the routing boundary is not free. On vLLM the
    call measured 426 us per layer per forward against a policy-free path that
    costs nothing, which is 26 ms of a ~1 s forward -- the whole reason `static`
    trailed the built-in selection it was supposed to reproduce. SGLang has no
    such gap because its `static` is one tensor index inside its own TopK code;
    declaring the property here lets a framework reach the same place without
    the pipeline having to know which runtime it is talking to.

    A policy that sets this must publish its answer through
    :attr:`PlacementSnapshot.default_physical_for_logical`.
    """

    consumes_global_logical_count: bool = False
    """Whether any policy reads the EP-wide per-logical token count.

    Distinct from ``requires_post_topk_routing``: a replica policy can run
    without ever looking at the load, and gathering the count for it costs a
    collective per layer per forward that is then discarded.

    This is the pipeline-level declaration a framework adapter reads once, at
    configuration time, to decide whether to gather the count at all. The
    orchestrator's own per-request, per-router check of the same underlying
    fact is :class:`~moe_load_balancer.core.l2.GlobalLogicalCountConsumer`.
    """

    # -- runtime properties the framework has to respect -------------------
    supports_concurrent_microbatches: bool = True
    """False when policies keep per-layer state that concurrent micro-batches
    would clobber."""

    graph_stability: str = "stable"
    """``stable``, or ``realloc_on_placement_change`` when a placement update
    can replace policy tensors and invalidate a captured graph."""


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
    placement_policy: Optional[str] = None
    """The L1 placement policy this L2 selection requires, if any.

    Set only when a replica policy's dispatch decision is meaningless without
    a specific placement algorithm's output -- ``ultraep``'s L2 router reads a
    quota table that only its own L1 policy produces. ``None`` for policies
    that work with whatever placement the framework already committed.
    """

    def is_applicable(self, topology: ExpertDeploymentConfig) -> Optional[str]:
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
        if "ultraep" in self.policies and topology.num_redundant_experts == 0:
            return (
                "ultraep places redundant replicas by quota, and this "
                "deployment has no redundant experts"
            )
        if "ultraep" in self.policies and not topology.carries_placement_metadata:
            return (
                "ultraep's L2 router reads the per-replica token quota its own "
                "L1 policy publishes on PlacementPlan.metadata, and this "
                "framework does not thread that metadata into the committed "
                "PlacementSnapshot -- without it every token routes to replica "
                "0 and the policy does nothing"
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
            # static decides from placement alone, so the framework may bake
            # its answer in at commit time instead of paying the boundary on
            # every forward.
            dispatch_fixed_by_placement=replica_policy == "static",
            routes_shared_expert=waterfill_policy is not None,
            requires_deepep=waterfill_policy is not None or replica_policy == "ultraep",
            consumes_global_logical_count=consumes_count,
            supports_concurrent_microbatches=replica_policy not in ("lplb", "ultraep"),
            graph_stability=(
                "realloc_on_placement_change"
                if replica_policy in ("lplb", "ultraep")
                else "stable"
            ),
        )
        return cls(
            name=canonical,
            policies=execution_policies,
            post_topk_policies=tuple(post_topk),
            input_transform=input_transform,
            capabilities=capabilities,
            placement_policy="ultraep" if replica_policy == "ultraep" else None,
        )
