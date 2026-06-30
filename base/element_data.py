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
    # terminal halides (monovalent anionic caps for the saturation stage)
    "F":  {"oxidation": -1, "max_cn": 1, "min_cn": 1},
    "Cl": {"oxidation": -1, "max_cn": 1, "min_cn": 1},
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


def _normalize_cn_variants(spec) -> list:
    """Normalize one element's CN distribution into a list of ``{cn, fraction[, oxidation]}``.

    Accepts a single int (the only expected CN), a ``{cn: fraction}`` mapping, or a list of
    ``{cn, fraction, oxidation?}`` dicts. Validates non-negative fractions summing to <= 1
    (any shortfall falls back to the element default CN)."""
    if isinstance(spec, bool):
        raise ValueError(f"cn_distr entry must be an int / mapping / list, not a bool: {spec!r}")
    if isinstance(spec, int):
        variants = [{"cn": int(spec), "fraction": 1.0}]
    elif isinstance(spec, dict):
        variants = [{"cn": int(cn), "fraction": float(fr)} for cn, fr in spec.items()]
    elif isinstance(spec, (list, tuple)):
        variants = []
        for entry in spec:
            if not isinstance(entry, dict) or "cn" not in entry or "fraction" not in entry:
                raise ValueError(f"cn_distr list entry must have 'cn' and 'fraction': {entry!r}")
            v = {"cn": int(entry["cn"]), "fraction": float(entry["fraction"])}
            if entry.get("oxidation") is not None:
                v["oxidation"] = int(entry["oxidation"])
            variants.append(v)
    else:
        raise ValueError(
            f"cn_distr entry must be an int, a {{cn: fraction}} mapping, or a list of "
            f"{{cn, fraction, oxidation}}; got {type(spec).__name__}")

    total = 0.0
    for v in variants:
        # A CN of 0 (or negative) would be stored as the per-atom max_cn override and then
        # read back as the "0 = unset" sentinel, silently reverting to the element default;
        # and a variant oxidation of 0 collides with the same unset sentinel. Reject both up
        # front so the distribution can't be silently ignored.
        if v["cn"] <= 0:
            raise ValueError(f"cn_distr: 'cn' must be a positive integer, got {v['cn']} in {variants!r}")
        if "oxidation" in v and v["oxidation"] == 0:
            raise ValueError("cn_distr: a variant 'oxidation' of 0 is not supported "
                             "(it collides with the unset sentinel); omit it to use the default")
        if v["fraction"] < 0:
            raise ValueError(f"cn_distr: negative fraction in {variants!r}")
        total += v["fraction"]
    if total > 1.0 + 1e-9:
        raise ValueError(f"cn_distr: fractions sum to {total:.3f} > 1.0 in {variants!r}")
    return variants


def normalize_cn_distr(cn_distr) -> dict:
    """Normalize a per-element CN distribution mapping to the canonical
    ``{element: [{cn, fraction[, oxidation]}, ...]}`` form. Idempotent, so it is safe to apply
    to already-normalized input (e.g. config-parsed) and to raw user input alike."""
    return {str(el): _normalize_cn_variants(spec) for el, spec in (cn_distr or {}).items()}


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
    # Validate the knobs so the coherence guarantees below cannot be silently broken by a
    # user-tuned 'distance' block: a negative bond_factor/bond_dev, a negative cutoff_pad
    # (which would put the cutoff below the window upper bound), or a collision_factor that is
    # not strictly less than bond_factor (which would make the clash radius reach into the
    # bond window) are all rejected here rather than producing incoherent tables.
    if bond_factor <= 0:
        raise ValueError(f"bond_factor must be > 0 (got {bond_factor})")
    if bond_dev < 0:
        raise ValueError(f"bond_dev must be >= 0 (got {bond_dev})")
    if cutoff_pad < 0:
        raise ValueError(f"cutoff_pad must be >= 0 (got {cutoff_pad}); the bonding cutoff would "
                         f"otherwise fall below the bond-sampling window")
    if collision_factor >= bond_factor:
        raise ValueError(f"collision_factor ({collision_factor}) must be < bond_factor "
                         f"({bond_factor}) so the collision floor stays below the bond window")

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
            # Coherence (enforced, not just asserted in tests): the different-element collision
            # floor must sit strictly below the bond window so a legitimately bonded neighbour
            # is never rejected as a clash. With collision_factor < bond_factor this holds for
            # large enough r; guard the small-radius pairs (e.g. H-H) explicitly.
            if dmin >= lo:
                raise ValueError(
                    f"collision floor for {a}-{b} (dmin={dmin:.3f}) is not below the bond "
                    f"window start (lo={lo:.3f}); lower collision_factor or bond_dev")
            windows[b] = uniform(loc=lo, scale=hi - lo)
            dmm[b] = [dmin, cutoff]
            pair_cutoffs[(a, b)] = cutoff
        sample_dist[a] = RandomSample(windows)
        d_min_max[a] = dmm
    return sample_dist, d_min_max, pair_cutoffs


def derive_bond_length(a: str, b: str, bond_factor: float | None = None) -> float:
    """Covalent-radii bond length for an ``a``-``b`` pair, in Angstrom.

    The same model used for the bond-sampling window centre in ``derive_pair_distances``
    (``bond_factor * (r_a + r_b)``), exposed standalone so the saturation stage can place
    caps at a coherent physical distance for *any* element pair -- including pairs the run's
    ``sample_dist`` never holds (it only carries grown cation-anion pairs, not e.g. X-H or
    X-F). ``bond_factor`` defaults to ``DEFAULT_DISTANCE_KNOBS['bond_factor']``.
    """
    if bond_factor is None:
        bond_factor = DEFAULT_DISTANCE_KNOBS["bond_factor"]
    return bond_factor * (_covalent_radius(a) + _covalent_radius(b))


def build_element_tables(
        elements,
        *,
        spectator_elements=None,
        distance_knobs: dict | None = None,
        oxidation: dict | None = None,
        max_cn: dict | None = None,
        min_cn: dict | None = None,
    ) -> dict:
    """Assemble the per-element and per-pair tables for ``elements``.

    Returns a dict with keys ``oxidation``, ``max_cn``, ``min_cn`` (per-element) and
    ``sample_dist``, ``d_min_max``, ``cut_offs`` (per-pair, from the covalent-radii
    derivation). Per-element values come from the curated table, with the optional
    ``oxidation``/``max_cn``/``min_cn`` override dicts taking precedence. An element absent
    from the curated table is allowed only if fully specified by the overrides; otherwise the
    clear missing-element error is raised.

    ``spectator_elements`` are present in the structure but never grown -- e.g. a loaded
    substrate of a foreign element. They get per-pair distance/cutoff rows (so placement and
    the coordination graph work for them) from covalent radii, but are NOT required to have a
    curated oxidation/CN and are omitted from the per-element tables (the per-atom CN arrays
    tolerate them).
    """
    elements = list(dict.fromkeys(elements))
    spectators = [e for e in dict.fromkeys(spectator_elements or []) if e not in elements]
    knobs = {**DEFAULT_DISTANCE_KNOBS, **(distance_knobs or {})}
    oxidation = oxidation or {}
    max_cn = max_cn or {}
    min_cn = min_cn or {}

    out_ox: dict = {}
    out_max: dict = {}
    out_min: dict = {}
    for e in elements:
        rec = OXIDE_ELEMENTS.get(e)
        if rec is None and not (e in oxidation and e in max_cn and e in min_cn):
            element_record(e)   # not curated and not fully overridden -> clear error
        rec = rec or {}
        out_ox[e] = oxidation.get(e, rec.get("oxidation"))
        out_max[e] = max_cn.get(e, rec.get("max_cn"))
        out_min[e] = min_cn.get(e, rec.get("min_cn"))

    # Distances cover the grown elements AND any spectators, so a loaded foreign substrate is a
    # valid collision/graph partner instead of triggering a KeyError during placement.
    sample_dist, d_min_max, cut_offs = derive_pair_distances(elements + spectators, **knobs)
    return {
        "oxidation": out_ox,
        "max_cn": out_max,
        "min_cn": out_min,
        "sample_dist": sample_dist,
        "d_min_max": d_min_max,
        "cut_offs": cut_offs,
    }
