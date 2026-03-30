from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from base import AmorphousStruc

class AlterationRule(ABC):
    """Abstract base class for all structural modification rules."""
    
    @abstractmethod
    def apply(self, atoms: AmorphousStruc, graph: nx.Graph) -> bool:
        """
        Applies the rule to the structure.
        Returns True if a modification was made, False otherwise.
        """
        pass