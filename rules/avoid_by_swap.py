from .alteration_rule import AlterationRule
from ase import Atoms
import networkx as nx
from typing import Optional

class AvoidMotifSwapRule(AlterationRule):
    def __init__(self, edge_element: str, center_element: str, swap_candidate: str, edge_element_2: Optional[str] = None):
        """
        Prevents edge-center-edge_2 motifs (e.g., Al-O-Ga).
        If edge_element_2 is not provided, defaults to edge_element (e.g., Al-O-Al).
        If found, swaps the first edge_element with a swap_candidate of the SAME degree.
        """
        self.edge = edge_element
        self.center = center_element
        # Default to the first edge element if a second one isn't specified
        self.edge_2 = edge_element_2 if edge_element_2 is not None else edge_element
        self.swap = swap_candidate

    def apply(self, atoms: Atoms, graph: nx.Graph) -> bool:
        edge_indices = [i for i, a in enumerate(atoms) if a.symbol == self.edge]
        
        for edge_idx in edge_indices:
            for neighbor in graph.neighbors(edge_idx):
                if atoms[neighbor].symbol == self.center:
                    # Check the center atom's neighbors for the second edge element
                    for next_neighbor in graph.neighbors(neighbor):
                        # Use self.edge_2 for the check here
                        if next_neighbor != edge_idx and atoms[next_neighbor].symbol == self.edge_2:
                            
                            # Motif found! 
                            # 1. Determine the degree (coordination) of the atom we want to swap out
                            target_degree = graph.degree[edge_idx]
                            
                            # 2. Find valid candidates of the correct element AND matching degree
                            swap_indices = [
                                i for i, a in enumerate(atoms) 
                                if a.symbol == self.swap and graph.degree[i] == target_degree
                            ]
                            
                            if not swap_indices:
                                continue # Cannot resolve if no candidates exist with the same degree
                            
                            target_swap_idx = swap_indices[0] 
                            
                            # Perform the swap
                            atoms[edge_idx].symbol = self.swap
                            atoms[target_swap_idx].symbol = self.edge
                            
                            return True # Return immediately to trigger graph rebuild
        return False