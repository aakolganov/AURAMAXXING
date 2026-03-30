from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Union

import numpy as np
from ase import Atoms, Atom
from ase.data import atomic_numbers
from numpy.typing import ArrayLike
from ase.neighborlist import neighbor_list
import networkx as nx
from collections import Counter
from default_constants import default_max_cn, pair_cutoffs, OXIDATION_POS, OXIDATION_NEG, default_min_cn
from .limits import Limits

@dataclass
class AmorphousStruc:
    atoms: Atoms
    max_cn: dict = field(default_factory=default_max_cn.copy)
    min_cn: dict = field(default_factory=default_min_cn.copy)
    cut_offs: dict = field(default_factory=pair_cutoffs.copy)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    frozen_indices: list[int] = field(default_factory=list)
    limits: Limits = field(default=None, init=False, repr=False)
    
    _graph: nx.Graph | None = field(default=None, init=False, repr=False)
    _need_graph_update: bool = field(default=True, init=False, repr=False)


    def __len__(self):
        return len(self.atoms)
    
    @property
    def has_frozen(self) -> bool:
        return len(self.frozen_indices) > 0

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
        if index is not None:
            # The degree of a node is its coordination number
            return self.get_graph().degree(index)
        
        # Return an array of CNs for all atoms in the structure
        return np.array([self.get_graph().degree(i) for i in range(len(self.atoms))])


    def get_atom_count(self, atom_type: str) -> int:
        """counts atoms of certain type"""
        return self.atoms.get_chemical_symbols().count(atom_type)
    

    def commit_atom(self, atom_type: str, position: np.ndarray) -> None:
        """Simple wrap to add an atom and trigger a graph update."""
        self.atoms.append(Atom(atom_type, position=position))
        self._need_graph_update = True


    def replace_atom(self, new_atom_type: str, new_position: np.ndarray, index: int) -> None:
        """Simple wrap to replace an atom and trigger a graph update."""
        self.atoms.positions[index] = new_position
        self.atoms.numbers[index] = atomic_numbers[new_atom_type]
        self._need_graph_update = True
        

    def remove_atom(self, index: Union[int, slice, List[int], np.ndarray]) -> None:
        """
        Remove an atom (or multiple atoms) and trigger a graph update.
        Handles updating frozen_indices if necessary.
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

        if self.has_frozen:
            # Map old indices to new indices
            idx_map = np.cumsum(keep_mask) - 1
            # Filter and map the frozen indices
            self.frozen_indices = [
                int(idx_map[i]) for i in self.frozen_indices if keep_mask[i]
            ]

        self.atoms = self.atoms[keep_mask]
        self._need_graph_update = True


    def sort_atoms(self) -> None:
        self.atoms = self.atoms[self.atoms.numbers.argsort()]


    def _rebuild_graph(self) -> None:
        """Create a new NetworkX graph using ASE's spatial neighbor list."""
        self._need_graph_update = False
        cutoffs = self._cutoff_list()

        # 1. Use ASE to efficiently find all interacting pairs
        # 'ij' returns two arrays: indices of the first atom, and indices of the second
        i_indices, j_indices = neighbor_list('ij', self.atoms, cutoffs)

        # 2. Initialize an undirected graph
        G = nx.Graph()

        # 3. Add nodes with metadata (useful for later structural rules)
        for idx, atom in enumerate(self.atoms):
            G.add_node(idx, symbol=atom.symbol)

        # 4. Add edges from the ASE neighbor pairs
        G.add_edges_from(zip(i_indices, j_indices))
        
        # Clean up any self-interactions (if periodic boundary conditions caused them)
        G.remove_edges_from(nx.selfloop_edges(G))

        self._graph = G


    def _cutoff_list(self) -> dict:
        """Return the element-pair cutoff dictionary."""
        return self.cut_offs


    def charge(self, defined_charges: Optional[dict[str, int]] = None) -> int:
        if defined_charges is None:
            defined_charges = {}
            [defined_charges.update(d) for d in [OXIDATION_NEG, OXIDATION_POS]]
        atom_counts = Counter(self.atoms.get_chemical_symbols())
        
        net_charge = 0
        for at, count in atom_counts.items():
            net_charge += count * defined_charges[at]
        return net_charge


def AmorphousStruc_factory(
    atoms: Optional[Atoms] = None,
    symbols: Optional[Sequence[str]] = None,
    positions: Optional[ArrayLike] = None,
    cell: Optional[ArrayLike] = None,
    pbc: Union[bool, Sequence[bool]] = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> AmorphousStruc:
    """
    Factory to create an AmorphousStruc object from either an existing
    ASE Atoms object or from its constituent parts.

    You can provide either:
      - atoms: An existing ASE Atoms object.
      - symbols & positions: To build a new Atoms object.

    If 'atoms' is provided, other arguments (symbols, positions, etc.) are ignored.
    If neither is provided, an empty structure is created.
    """
    # If the user passes in a Generator, use it; otherwise build one from the seed.
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    if not cell is None:
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

    return AmorphousStruc(atoms=atoms_obj, rng=rng)