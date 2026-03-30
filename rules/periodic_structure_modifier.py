from .alteration_rule import AlterationRule
from base import AmorphousStruc
import networkx as nx

class PeriodicStructureModifier:
    def __init__(self, amorphous_struct: AmorphousStruc):
        self.atoms = amorphous_struct.atoms
        self.build_graph = amorphous_struct.get_graph
        self.graph: nx.Graph = self.build_graph(force_rebuild=True)
        self.rules: list[AlterationRule] = []

    def add_rule(self, rule: AlterationRule):
        """Adds a rule to the optimization pipeline."""
        self.rules.append(rule)

    def optimize(self, max_iterations=100) -> bool:
        """
        Runs all rules iteratively until no changes are made or max_iterations is reached.
        Returns True if fully converged.
        """
        for iteration in range(max_iterations):
            any_changes_this_pass = False
            
            for rule in self.rules:
                if rule.apply(self.atoms, self.graph):
                    any_changes_this_pass = True
                    # A change occurred, rebuild the graph to keep topology in sync
                    self.graph = self.build_graph(force_rebuild=True)
            
            if not any_changes_this_pass:
                print(f"Structure converged after {iteration} iterations.")
                return True
                
        print(f"Warning: Reached {max_iterations} iterations without convergence.")
        return False