from .alteration_rule import AlterationRule
import numpy as np
from typing import Optional
from ase import Atoms
from ase.neighborlist import neighbor_list
import networkx as nx

class MinimumDistanceRule(AlterationRule):
    def __init__(self, min_dist: float, element1: Optional[str]=None, element2: Optional[str]=None, step_size:float=0.1):
        """
        Enforces a minimum distance between element1 and element2.
        If element1 or 2 are None, it applies to all atoms.
        """
        self.min_dist = min_dist
        self.e1 = element1
        self.e2 = element2
        self.step = step_size

    def apply(self, atoms: Atoms, graph: nx.Graph) -> bool:
        # O(N) neighbour search for pairs closer than min_dist, instead of building
        # the full O(N^2) distance matrix. neighbor_list returns each violating pair
        # in both directions; we act on the first one matching the element filter.
        i_idx, j_idx = neighbor_list("ij", atoms, self.min_dist)

        for idx_a, idx_b in zip(i_idx, j_idx, strict=True):
            sym_a, sym_b = atoms[idx_a].symbol, atoms[idx_b].symbol
            
            # Check if this pair matches our target elements
            match_1 = (self.e1 is None or sym_a == self.e1) and (self.e2 is None or sym_b == self.e2)
            match_2 = (self.e1 is None or sym_b == self.e1) and (self.e2 is None or sym_a == self.e2)
            
            if match_1 or match_2:
                # vec points from A to B (r_b - r_a); push the two atoms apart
                # symmetrically, half a step each.
                vec = atoms.get_distance(idx_a, idx_b, vector=True, mic=True)
                norm = np.linalg.norm(vec)
                if norm < 1e-9:
                    # Coincident atoms: pick an arbitrary fixed direction to separate them.
                    vec = np.array([1.0, 0.0, 0.0])
                    norm = 1.0

                half_step = (vec / norm) * (self.step / 2.0)
                atoms.positions[idx_a] -= half_step  # A moves away from B
                atoms.positions[idx_b] += half_step  # B moves away from A
                return True
                
        return False