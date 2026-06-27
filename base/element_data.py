"""Data-driven element defaults for arbitrary oxide compositions.

ASE ships ``covalent_radii`` but no oxidation-state or coordination-number tables, so
those defaults live here as a small curated table (``OXIDE_ELEMENTS``) for common oxide
formers. ``build_element_tables`` assembles, for the element set of a run, the per-element
(oxidation / max_cn / min_cn) and per-pair (bond-sampling / collision / bonding-cutoff)
dicts that the rest of the code already consumes -- so growth/placement/picker keep eating
the same dict shapes, only now built dynamically instead of hand-written for Si/Al/O/H.

Distances come from a single covalent-radii model (``derive_pair_distances``) shared by the
sampling window, the collision floor and the graph-edge cutoff. Deriving all three from the
same basis makes them coherent *by construction*: a committed bond is always inside the
bonding cutoff (so it is always counted as a graph edge), and a legitimately bonded
neighbour is never rejected as a steric clash.

Physical caveat: the covalent-radii sum overshoots short, partly-ionic oxide bonds (e.g.
Si-O is ~1.61 A but covalent_radii[Si]+covalent_radii[O] ~ 1.77 A). The generator only needs
roughly-bonded, collision-free placements; the post-growth relaxation (MLIP/BKS) pulls the
geometry to true bond lengths. The model factors below are tunable per run.
"""
from ase.data import atomic_numbers, covalent_radii
from scipy.stats import uniform

from helpers.random_sample import RandomSample


# Curated oxide-former defaults. ``oxidation`` is the signed formal charge (its sign defines
# cation vs anion). ``max_cn``/``min_cn`` are typical oxide coordination numbers; ``min_cn``
# is the under-coordination floor used by saturation, ``max_cn`` the growth saturation cap.
# The Si/Al/O/H rows reproduce the legacy default_constants tables exactly. Multivalent
# elements use their most common oxide oxidation state; override per run when needed.
OXIDE_ELEMENTS: dict[str, dict] = {
    # anion + hydrogen (match legacy OXIDATION_NEG / default_*_cn)
    "O":  {"oxidation": -2, "max_cn": 2, "min_cn": 2},
    "H":  {"oxidation": +1, "max_cn": 1, "min_cn": 1},
    # legacy cations (match OXIDATION_POS / default_*_cn exactly)
    "Si": {"oxidation": +4, "max_cn": 4, "min_cn": 4},
    "Al": {"oxidation": +3, "max_cn": 4, "min_cn": 3},
    # common oxide formers
    "B":  {"oxidation": +3, "max_cn": 3, "min_cn": 3},
    "P":  {"oxidation": +5, "max_cn": 4, "min_cn": 4},
    "Ge": {"oxidation": +4, "max_cn": 4, "min_cn": 4},
    "Ga": {"oxidation": +3, "max_cn": 4, "min_cn": 4},
    "Ti": {"oxidation": +4, "max_cn": 6, "min_cn": 4},
    "Zr": {"oxidation": +4, "max_cn": 7, "min_cn": 6},
    "Hf": {"oxidation": +4, "max_cn": 7, "min_cn": 6},
    "Sn": {"oxidation": +4, "max_cn": 6, "min_cn": 4},
    "Zn": {"oxidation": +2, "max_cn": 4, "min_cn": 4},
    "Mg": {"oxidation": +2, "max_cn": 6, "min_cn": 4},
    "Ca": {"oxidation": +2, "max_cn": 6, "min_cn": 6},
    "Sr": {"oxidation": +2, "max_cn": 8, "min_cn": 6},
    "Ba": {"oxidation": +2, "max_cn": 8, "min_cn": 6},
    "Be": {"oxidation": +2, "max_cn": 4, "min_cn": 4},
    "Li": {"oxidation": +1, "max_cn": 4, "min_cn": 4},
    "Na": {"oxidation": +1, "max_cn": 6, "min_cn": 4},
    "K":  {"oxidation": +1, "max_cn": 8, "min_cn": 6},
    "Sc": {"oxidation": +3, "max_cn": 6, "min_cn": 6},
    "Y":  {"oxidation": +3, "max_cn": 6, "min_cn": 6},
    "La": {"oxidation": +3, "max_cn": 7, "min_cn": 6},
    "Nb": {"oxidation": +5, "max_cn": 6, "min_cn": 6},
    "Ta": {"oxidation": +5, "max_cn": 6, "min_cn": 6},
    "W":  {"oxidation": +6, "max_cn": 6, "min_cn": 6},
    # multivalent: most common oxide state as default (override per run if needed)
    "Fe": {"oxidation": +3, "max_cn": 6, "min_cn": 4},
    "Cr": {"oxidation": +3, "max_cn": 6, "min_cn": 4},
    "V":  {"oxidation": +5, "max_cn": 6, "min_cn": 4},
    "Mo": {"oxidation": +6, "max_cn": 6, "min_cn": 4},
    "Ce": {"oxidation": +4, "max_cn": 8, "min_cn": 6},
}


# Covalent-radii distance-model defaults. ``bond_factor`` scales the covalent-radii sum to
# the sampling-window centre; ``bond_dev`` is the +/- half-width (A); the graph-edge cutoff
# is the window upper bound + ``cutoff_pad`` (so a sampled bond is always within the cutoff);
# the different-element collision floor is ``collision_factor`` * sum (kept below the window
# lower bound so a bonded neighbour is never a clash). See the module docstring on "1.2x":
# here the bond centre is ~1.0x the covalent-radii sum (matching real oxide geometry and the
# legacy silica placement) and the cutoff sits above it, rather than placing bonds at 1.2x.
DEFAULT_DISTANCE_KNOBS: dict[str, float] = {
    "bond_factor": 1.0,
    "bond_dev": 0.15,
    "cutoff_pad": 0.25,
    "collision_factor": 0.75,
}


def element_record(symbol: str) -> dict:
    """Curated defaults for ``symbol`` (oxidation/max_cn/min_cn). Raises a clear error,
    naming the element, when it is not in the curated table -- the caller is expected to add
    it here or supply its values explicitly in the run config."""
    try:
        return OXIDE_ELEMENTS[symbol]
    except KeyError:
        raise ValueError(
            f"element {symbol!r} is not in the curated oxide table "
            f"(base/element_data.py:OXIDE_ELEMENTS). Add it there, or supply its oxidation "
            f"and max_cn/min_cn explicitly in the run config's 'coordination' block."
        ) from None


def _covalent_radius(symbol: str) -> float:
    """ASE covalent radius for ``symbol`` (A), with a clear error for an unknown or
    radius-less element instead of an opaque index/zero downstream."""
    z = atomic_numbers.get(symbol)
    if z is None:
        raise ValueError(f"unknown element symbol {symbol!r} (not in ase.data.atomic_numbers)")
    r = float(covalent_radii[z])
    if not r > 0:
        raise ValueError(f"no covalent radius for {symbol!r} in ase.data.covalent_radii")
    return r


def derive_pair_distances(
        elements,
        *,
        bond_factor: float = 1.0,
        bond_dev: float = 0.15,
        cutoff_pad: float = 0.25,
        collision_factor: float = 0.75,
    ) -> tuple[dict, dict, dict]:
    """Derive coherent per-pair distance tables from ASE covalent radii.

    Returns ``(sample_dist, d_min_max, pair_cutoffs)`` over every ordered pair of
    ``elements`` -- the exact shapes the existing consumers index:

    - ``sample_dist[a][b]`` is a ``RandomSample`` of ``scipy.stats.uniform`` over the bond
      window ``[lo, hi)`` with ``centre = bond_factor*(r_a+r_b)``, ``lo = centre-bond_dev``,
      ``hi = centre+bond_dev``. Accessing it draws a sample (RandomSample.__getitem__).
    - ``pair_cutoffs[(a,b)] = hi + cutoff_pad`` (symmetric). Since ``uniform`` support is the
      half-open ``[lo, hi)``, every draw is ``< hi <= cutoff`` -- a committed bond is always a
      graph edge (the coherence guarantee).
    - ``d_min_max[a][b] = [dmin, dmax]`` with ``dmin = collision_factor*(r_a+r_b)`` (the
      different-element exclusion radius, kept below ``lo``) and ``dmax = cutoff`` (the
      same-element exclusion radius == the self-bond cutoff, blocking spurious homo bonds).
    """
    elements = list(dict.fromkeys(elements))   # dedup, preserve order
    radii = {e: _covalent_radius(e) for e in elements}

    sample_dist: dict = {}
    d_min_max: dict = {}
    pair_cutoffs: dict = {}
    for a in elements:
        windows: dict = {}
        dmm: dict = {}
        for b in elements:
            r = radii[a] + radii[b]
            center = bond_factor * r
            lo = center - bond_dev
            hi = center + bond_dev
            if lo < 0:
                raise ValueError(
                    f"derived bond window for {a}-{b} starts below 0 (lo={lo:.3f}); "
                    f"reduce bond_dev or raise bond_factor")
            cutoff = hi + cutoff_pad
            dmin = collision_factor * r
            windows[b] = uniform(loc=lo, scale=hi - lo)
            dmm[b] = [dmin, cutoff]
            pair_cutoffs[(a, b)] = cutoff
        sample_dist[a] = RandomSample(windows)
        d_min_max[a] = dmm
    return sample_dist, d_min_max, pair_cutoffs


def build_element_tables(elements, *, distance_knobs: dict | None = None) -> dict:
    """Assemble the per-element and per-pair default tables for ``elements``.

    Returns a dict with keys ``oxidation``, ``max_cn``, ``min_cn`` (per-element, from the
    curated table) and ``sample_dist``, ``d_min_max``, ``cut_offs`` (per-pair, from the
    covalent-radii derivation). User overrides are applied by the caller (the config layer),
    mirroring the existing partial-merge-onto-defaults convention.
    """
    elements = list(dict.fromkeys(elements))
    knobs = {**DEFAULT_DISTANCE_KNOBS, **(distance_knobs or {})}

    oxidation: dict = {}
    max_cn: dict = {}
    min_cn: dict = {}
    for e in elements:
        rec = element_record(e)
        oxidation[e] = rec["oxidation"]
        max_cn[e] = rec["max_cn"]
        min_cn[e] = rec["min_cn"]

    sample_dist, d_min_max, cut_offs = derive_pair_distances(elements, **knobs)
    return {
        "oxidation": oxidation,
        "max_cn": max_cn,
        "min_cn": min_cn,
        "sample_dist": sample_dist,
        "d_min_max": d_min_max,
        "cut_offs": cut_offs,
    }
