from .alteration_rule import AlterationRule
import numpy as np
from typing import Optional
from ase import Atoms
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
        dm = atoms.get_all_distances(mic=True)
        # Create a mask to ignore self-interactions
        np.fill_diagonal(dm, np.inf) 
        
        # Find where distance is too small
        close_pairs = np.argwhere(dm < self.min_dist)
        
        for idx_a, idx_b in close_pairs:
            sym_a, sym_b = atoms[idx_a].symbol, atoms[idx_b].symbol
            
            # Check if this pair matches our target elements
            match_1 = (self.e1 is None or sym_a == self.e1) and (self.e2 is None or sym_b == self.e2)
            match_2 = (self.e1 is None or sym_b == self.e1) and (self.e2 is None or sym_a == self.e2)
            
            if match_1 or match_2:
                # Calculate vector and push apart
                vec = atoms.get_distance(idx_a, idx_b, vector=True, mic=True)
                norm = np.linalg.norm(vec)
                
                # Nudge atom A away from atom B
                push_vector = (vec / norm) * self.step
                atoms.positions[idx_a] += push_vector
                return True
                
        return False