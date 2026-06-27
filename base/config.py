"""User-adjustable structural configuration for an AmorphousStruc.

Bundles the coordination-related knobs that were previously only settable by
hand-constructing the dataclass, so they can be passed through the factory and the
initialize_structure_* helpers as a single object. Every field defaults to the
current module-level constant, so omitting the config reproduces today's behaviour.
"""
from dataclasses import dataclass, field

from default_constants import (
    default_max_cn,
    default_min_cn,
    pair_cutoffs,
    default_overcoord_policy,
    default_oxidation,
)


@dataclass
class CoordinationConfig:
    """Coordination limits and the overcoordination policy for a structure.

    Parameters
    ----------
    max_cn, min_cn : dict
        Per-element maximum / minimum coordination numbers.
    cut_offs : dict
        Element-pair bonding cutoffs used to build the coordination graph.
    overcoord_policy : dict
        Per-element overcoordination policy, e.g. ``{"Al": {"max_cn": 6,
        "fraction": 0.2}}``. ``{}`` (default) disables the feature.
    oxidation : dict
        Per-element signed oxidation state (sign defines cation vs anion). Used by
        ``charge()`` and the growth attachment rule.
    """
    max_cn: dict = field(default_factory=default_max_cn.copy)
    min_cn: dict = field(default_factory=default_min_cn.copy)
    cut_offs: dict = field(default_factory=pair_cutoffs.copy)
    overcoord_policy: dict = field(default_factory=default_overcoord_policy.copy)
    oxidation: dict = field(default_factory=default_oxidation.copy)
