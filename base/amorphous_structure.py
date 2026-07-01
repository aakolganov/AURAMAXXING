from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Union

import numpy as np
from ase import Atoms, Atom
from ase.data import atomic_numbers
from numpy.typing import ArrayLike
from ase.neighborlist import neighbor_list
import networkx as nx
from collections import Counter
from default_constants import (
    default_max_cn,
    pair_cutoffs,
    default_oxidation,
    default_min_cn,
    default_cn_distr,
    sample_dist as default_sample_dist,
    d_min_max as default_d_min_max,
)
from .element_data import normalize_cn_distr
from .limits import Limits
from .config import CoordinationConfig

@dataclass
class AmorphousStruc:
    atoms: Atoms
    max_cn: dict = field(default_factory=default_max_cn.copy)
    min_cn: dict = field(default_factory=default_min_cn.copy)
    cut_offs: dict = field(default_factory=pair_cutoffs.copy)
    sample_dist: dict = field(default_factory=default_sample_dist.copy)
    d_min_max: dict = field(default_factory=default_d_min_max.copy)
    cn_distr: dict = field(default_factory=default_cn_distr.copy)
    oxidation: dict = field(default_factory=default_oxidation.copy)
    # Covalent-radii bond-window scale used to place saturation caps at the run's bond scale
    # (defaults to 1.0x, matching the legacy/default placement).
    bond_factor: float = 1.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    limits: Limits = field(default=None, init=False, repr=False)

    _graph: nx.Graph | None = field(default=None, init=False, repr=False)
    _need_graph_update: bool = field(default=True, init=False, repr=False)

    def __post_init__(self):
        # Accept any supported cn_distr input form (int / {cn: fraction} / list of dicts) and
        # store the canonical normalized list form for the sampler. Idempotent.
        self.cn_distr = normalize_cn_distr(self.cn_distr)


    def __len__(self):
        return len(self.atoms)

    @property
    def symbols(self) -> list[str]:
        return self.atoms.get_chemical_symbols()
    

    def set_seed(self, seed: Optional[Union[int, np.random.Generator]]) -> None:
        """Resets the random number generator with a new seed."""
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

    def get_graph(self, force_rebuild: bool = False) -> nx.Graph:
        if self._need_graph_update or force_rebuild:
            self._rebuild_graph()
        return self._graph


    def invalidate_graph(self) -> None:
        """Mark the cached coordination graph stale so the next ``get_graph``/``get_cn``
        rebuilds it from the current positions.

        The cache is kept in sync automatically for the class's own mutators
        (``commit_atom``/``replace_atom``/``remove_atom``). Call this after positions are
        changed *outside* those methods -- e.g. an ASE optimizer or MD run relaxes
        ``atoms.positions`` in place, which the cache cannot otherwise detect.
        """
        self._need_graph_update = True


    def get_cn(self, index: Optional[int] = None) -> Union[int, np.ndarray]:
        """
        Get coordination number (CN) of one or all atoms.

        Parameters
        ----------
        index : int, optional
            If given, return CN for this atom index. If None, return an
            array of CNs for all atoms.

        Returns
        -------
        int | np.ndarray
            The coordination number(s).
        """
        graph = self.get_graph()
        if index is not None:
            # The degree of a node is its coordination number
            return graph.degree(index)

        # Return an array of CNs for all atoms. dict(graph.degree()) materialises
        # every degree in a single pass; the previous per-node graph.degree(i) cost
        # ~two networkx calls per atom and dominated the growth hot path.
        degrees = dict(graph.degree())
        return np.array([degrees[i] for i in range(len(self.atoms))])


    def max_cn_array(self) -> np.ndarray:
        """Per-atom maximum coordination number.

        Combines the per-element ``max_cn`` defaults with any per-atom overrides
        stored in ``atoms.arrays["max_cn"]`` (set from ``cn_distr``).
        A stored value of 0 means "unset" and falls back to the element default, so
        ASE's zero-padding of freshly appended atoms is harmless. When no overrides
        exist the result is exactly the per-element defaults (back-compatible).
        """
        symbols = np.array(self.symbols)
        # A foreign element with no max_cn (e.g. a loaded substrate of an unconfigured
        # element) is treated as already saturated (huge cap) so growth never tries to
        # attach to it, rather than crashing with a KeyError.
        base = np.array([self.max_cn.get(s, 10**6) for s in symbols], dtype=int)
        override = self.atoms.arrays.get("max_cn")
        if override is None:
            return base
        return np.where(override > 0, override, base)


    def min_cn_array(self) -> np.ndarray:
        """Per-atom minimum coordination number (per-element only).

        ``cn_distr`` sets the per-atom upper (max) coordination only; the
        under-coordination floor that drives saturation stays per-element.
        """
        symbols = np.array(self.symbols)
        # A foreign element with no min_cn is never flagged under-coordinated (floor 0).
        return np.array([self.min_cn.get(s, 0) for s in symbols], dtype=int)


    def _assign_cn_variant(self, symbol: str) -> tuple[int, int]:
        """Assign a new atom's coordination number (and oxidation) from the element's
        ``cn_distr``.

        Returns ``(max_cn_tag, oxidation_tag)``; 0 means "unset, use the element default".
        Draws one categorical sample over the distribution's variants; the CN and oxidation
        come from the SAME draw, so a variant atom gets both together. Short-circuits with NO
        rng draw when the element has no distribution -- this keeps distribution-free runs
        byte-for-byte identical. A draw beyond the variants' cumulative fraction (sum < 1)
        falls back to the element default.
        """
        variants = self.cn_distr.get(symbol)
        if not variants:
            return 0, 0
        u = self.rng.random()
        cum = 0.0
        for v in variants:
            cum += v["fraction"]
            if u < cum:
                return int(v["cn"]), int(v.get("oxidation", 0))
        return 0, 0


    @property
    def _distr_has_oxidation(self) -> bool:
        """True when any cn_distr variant carries an ``oxidation`` -- only then is the per-atom
        ``oxidation`` array materialised, so distributions without it stay byte-for-byte as
        before."""
        return any("oxidation" in v
                   for variants in self.cn_distr.values() for v in variants)


    def apply_cn_distr(self) -> None:
        """(Re)assign per-atom coordination (and oxidation) overrides for all current atoms
        from ``cn_distr`` and store them in ``atoms.arrays``. No-op when ``cn_distr`` is empty.
        Use this to opt a loaded structure into the distribution after seeding (``set_seed``);
        grown atoms are assigned incrementally in ``commit_atom`` instead.
        """
        if not self.cn_distr:
            return
        pairs = [self._assign_cn_variant(s) for s in self.symbols]
        self.atoms.set_array("max_cn", np.array([p[0] for p in pairs], dtype=int))
        if self._distr_has_oxidation:
            self.atoms.set_array("oxidation", np.array([p[1] for p in pairs], dtype=int))


    def get_atom_count(self, atom_type: str) -> int:
        """counts atoms of certain type"""
        return self.atoms.get_chemical_symbols().count(atom_type)
    

    def commit_atom(self, atom_type: str, position: np.ndarray) -> None:
        """Add an atom and keep the coordination graph in sync.

        Appending only adds edges for the new atom, so when an up-to-date graph
        already exists we extend it incrementally instead of forcing a full
        O(N) neighbour-list rebuild on the next query (the growth hot path adds
        one atom at a time). Otherwise we just mark the graph stale.
        """
        self.atoms.append(Atom(atom_type, position=position))
        new_idx = len(self.atoms) - 1

        # Assign the new atom's CN (and oxidation) from cn_distr. ASE pads an existing per-atom
        # array with 0 on append, so we overwrite that slot; if the array doesn't exist yet
        # (e.g. first atom of a blank struct) we create it. Guarded by a non-empty cn_distr so
        # distribution-off runs are unchanged.
        if self.cn_distr:
            mx_tag, ox_tag = self._assign_cn_variant(atom_type)
            if "max_cn" not in self.atoms.arrays:
                self.atoms.set_array("max_cn", np.zeros(len(self.atoms), dtype=int))
            self.atoms.arrays["max_cn"][new_idx] = mx_tag
            # The variant oxidation comes from the same draw; only materialised when a
            # distribution uses 'oxidation', so CN-only distributions are unchanged.
            if self._distr_has_oxidation:
                if "oxidation" not in self.atoms.arrays:
                    self.atoms.set_array("oxidation", np.zeros(len(self.atoms), dtype=int))
                self.atoms.arrays["oxidation"][new_idx] = ox_tag

        if self._graph is not None and not self._need_graph_update:
            self._add_atom_to_graph(new_idx)
        else:
            self._need_graph_update = True

    def _add_atom_to_graph(self, idx: int) -> None:
        """Add node `idx` (the last atom) and its edges to the cached graph.

        Mirrors _rebuild_graph's edge rule exactly: an edge is added when the
        minimum-image distance is strictly below the element-pair cutoff.
        """
        self._graph.add_node(idx, symbol=self.atoms[idx].symbol)
        if idx == 0:
            return
        sym_new = self.atoms[idx].symbol
        symbols = self.atoms.get_chemical_symbols()
        dists = self.atoms.get_distances(idx, list(range(idx)), mic=True)
        for j, d in enumerate(dists):
            # Order-independent lookup: ASE's neighbor_list (used by the full rebuild) treats
            # element pairs as unordered, so try both key orders to match it exactly even when
            # cut_offs is not symmetric (#13).
            cutoff = self.cut_offs.get((sym_new, symbols[j]))
            if cutoff is None:
                cutoff = self.cut_offs.get((symbols[j], sym_new))
            if cutoff is not None and d < cutoff:
                self._graph.add_edge(idx, j)


    def replace_atom(self, new_atom_type: str, new_position: np.ndarray, index: int) -> None:
        """Simple wrap to replace an atom and trigger a graph update."""
        # If the element changes, drop any per-atom max-CN override so a stale tag
        # (e.g. an Al allowed CN 6) can't carry over to a different element. 0 falls
        # back to the new element's default. (Today move_atom only replaces in place
        # with the same symbol, so this is defensive.)
        if self.atoms[index].symbol != new_atom_type:
            for key in ("max_cn", "oxidation"):
                arr = self.atoms.arrays.get(key)
                if arr is not None:
                    arr[index] = 0
        self.atoms.positions[index] = new_position
        self.atoms.numbers[index] = atomic_numbers[new_atom_type]
        self._need_graph_update = True
        

    def remove_atom(self, index: Union[int, slice, List[int], np.ndarray]) -> None:
        """
        Remove an atom (or multiple atoms) and trigger a graph update.

        Any ASE constraints on the atoms (e.g. a FixAtoms substrate carried in
        from a VASP POSCAR with Selective Dynamics) are preserved and reindexed
        automatically by ASE's atom slicing below.
        """
        n_atoms = len(self.atoms)
        keep_mask = np.ones(n_atoms, dtype=bool)

        if isinstance(index, int):
            keep_mask[index] = False
        elif isinstance(index, slice):
            indices = np.arange(n_atoms)[index]
            keep_mask[indices] = False
        else:
            index = np.asarray(index)
            if index.dtype == bool:
                if len(index) != n_atoms:
                    raise ValueError(f"Boolean mask length {len(index)} != number of atoms {n_atoms}")
                keep_mask = ~index
            else:
                keep_mask[index] = False

        self.atoms = self.atoms[keep_mask]
        self._need_graph_update = True


    def sort_atoms(self) -> None:
        self.atoms = self.atoms[self.atoms.numbers.argsort()]
        # Reordering changes every node's index, so the cached graph (keyed by the old
        # indices) no longer matches the atoms; force a rebuild on the next query.
        self.invalidate_graph()


    def _rebuild_graph(self) -> None:
        """Create a new NetworkX graph using ASE's spatial neighbor list."""
        self._need_graph_update = False
        cutoffs = self.cut_offs

        # 1. Use ASE to efficiently find all interacting pairs
        # 'ij' returns two arrays: indices of the first atom, and indices of the second
        i_indices, j_indices = neighbor_list('ij', self.atoms, cutoffs)

        # 2. Initialize an undirected graph
        G = nx.Graph()

        # 3. Add nodes with metadata (useful for later structural rules)
        for idx, atom in enumerate(self.atoms):
            G.add_node(idx, symbol=atom.symbol)

        # 4. Add edges from the ASE neighbor pairs
        G.add_edges_from(zip(i_indices, j_indices, strict=False))
        
        # Clean up any self-interactions (if periodic boundary conditions caused them)
        G.remove_edges_from(nx.selfloop_edges(G))

        self._graph = G


    def charge(self, defined_charges: Optional[dict[str, int]] = None) -> int:
        """Net formal charge from the per-element oxidation table (``self.oxidation`` by
        default, or an explicit ``defined_charges`` override). Raises a clear error, naming
        the element, when an element present in the structure has no oxidation state."""
        table = self.oxidation if defined_charges is None else defined_charges
        symbols = self.atoms.get_chemical_symbols()

        def _element_charge(at: str) -> int:
            try:
                return table[at]
            except KeyError:
                raise ValueError(
                    f"charge(): no oxidation state for element {at!r}; add it to the "
                    f"structure's oxidation table (or the curated base/element_data table)."
                ) from None

        # Per-atom oxidation overrides (minority variants assigned by cn_distr) are
        # honoured only against the structure's own table; an explicit defined_charges applies
        # uniformly. 0 in the override array means "unset -> use the element default".
        override = None if defined_charges is not None else self.atoms.arrays.get("oxidation")
        if override is None:
            # Fast path (no variants): per-element sum, unchanged behaviour.
            atom_counts = Counter(symbols)
            return sum(count * _element_charge(at) for at, count in atom_counts.items())

        net_charge = 0
        for i, at in enumerate(symbols):
            q = int(override[i])
            net_charge += q if q != 0 else _element_charge(at)
        return net_charge


    def fixed_indices(self) -> set:
        """Indices held fixed by any ASE constraint (e.g. a FixAtoms substrate carried in
        from a VASP POSCAR's Selective Dynamics)."""
        fixed: set = set()
        for constraint in self.atoms.constraints:
            if hasattr(constraint, "get_indices"):
                fixed.update(int(i) for i in constraint.get_indices())
        return fixed


def AmorphousStruc_factory(
    atoms: Optional[Atoms] = None,
    symbols: Optional[Sequence[str]] = None,
    positions: Optional[ArrayLike] = None,
    cell: Optional[ArrayLike] = None,
    pbc: Union[bool, Sequence[bool]] = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
    config: Optional[CoordinationConfig] = None,
) -> AmorphousStruc:
    """
    Factory to create an AmorphousStruc object from either an existing
    ASE Atoms object or from its constituent parts.

    You can provide either:
      - atoms: An existing ASE Atoms object.
      - symbols & positions: To build a new Atoms object.

    If 'atoms' is provided, other arguments (symbols, positions, etc.) are ignored.
    If neither is provided, an empty structure is created.

    `config` (CoordinationConfig) sets the coordination limits, cutoffs and the
    coordination-number distribution (cn_distr). When omitted, the AmorphousStruc field
    defaults apply, reproducing the previous behaviour.
    """
    # If the user passes in a Generator, use it; otherwise build one from the seed.
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    if cell is not None:
        pbc = True
    if pbc and cell is None:
        raise ValueError("Cannot have periodic boundary conditions without a cell.")

    if atoms is not None:
        # Use a copy to avoid modifying the original object
        atoms_obj = atoms.copy()
    elif symbols is not None and positions is not None:
        atoms_obj = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
    else:
        # Create an empty Atoms object if no inputs are given
        atoms_obj = Atoms(cell=cell, pbc=pbc)

    kwargs = {}
    if config is not None:
        kwargs.update(
            max_cn=config.max_cn.copy(),
            min_cn=config.min_cn.copy(),
            cut_offs=config.cut_offs.copy(),
            sample_dist=config.sample_dist.copy(),
            d_min_max=config.d_min_max.copy(),
            cn_distr=config.cn_distr.copy(),
            oxidation=config.oxidation.copy(),
            bond_factor=config.bond_factor,
        )

    return AmorphousStruc(atoms=atoms_obj, rng=rng, **kwargs)