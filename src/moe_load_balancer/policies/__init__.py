"""Algorithm policy packages.

The L1/L2 sub-packages avoid eager imports so dev environments without the
heavy GPU dependencies (``torch`` / ``triton``) can still consume the
framework-neutral core contracts. Import concrete policies from their
sub-modules directly, e.g.
``from moe_load_balancer.policies.l2.waterfill import WaterfillL2Router``.
"""

__all__: list[str] = []
