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
    default_cn_distr,
    default_oxidation,
    sample_dist as default_sample_dist,
    d_min_max as default_d_min_max,
)


@dataclass
class CoordinationConfig:
    """Coordination limits and the coordination-number distribution for a structure.

    Parameters
    ----------
    max_cn, min_cn : dict
        Per-element maximum / minimum coordination numbers.
    cut_offs : dict
        Element-pair bonding cutoffs used to build the coordination graph.
    sample_dist, d_min_max : dict
        Per-pair bond-length sampling distributions and collision [min, max] ranges.
    cn_distr : dict
        Per-element expected coordination-number distribution: a single CN, a
        ``{cn: fraction}`` mapping, or a list of ``{cn, fraction, oxidation?}``. ``{}``
        (default) uses the curated per-element default CN for every element.
    oxidation : dict
        Per-element signed oxidation state (sign defines cation vs anion). Used by
        ``charge()`` and the growth attachment rule.
    """
    max_cn: dict = field(default_factory=default_max_cn.copy)
    min_cn: dict = field(default_factory=default_min_cn.copy)
    cut_offs: dict = field(default_factory=pair_cutoffs.copy)
    sample_dist: dict = field(default_factory=default_sample_dist.copy)
    d_min_max: dict = field(default_factory=default_d_min_max.copy)
    cn_distr: dict = field(default_factory=default_cn_distr.copy)
    oxidation: dict = field(default_factory=default_oxidation.copy)
    # Covalent-radii bond-window scale (centre = bond_factor*(r_a+r_b)); the saturation stage
    # uses it so caps are placed at the same scale as the grown network, not always 1.0x.
    bond_factor: float = 1.0
