"""L3 expert-weight-transfer policy implementations live here.

The actual ``UltraEPWeightTransfer`` import lives at
``moe_load_balancer.policies.l3.ultraep`` and defers importing ``ultra_ep``
(an optional third-party dependency) to inside
``UltraEPWeightTransfer.register_weights``, so a dev environment without it
can still consume the framework-neutral ``core.l3.ExpertWeightTransfer``
protocol. Import concrete policies from their sub-module directly, e.g.
``from moe_load_balancer.policies.l3.ultraep import UltraEPWeightTransfer``.
"""

__all__: list[str] = []
