import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

class SoftCoreRepulsion(Calculator):
    """
    Classic soft-core repulsive calculator (as used in DPD conservative forces).
    Only acts between specified atom type pairs.

    Potential: V(r) = 0.5 * A * rc * (1 - r/rc)^2
    Force:     F(r) = A * (1 - r/rc) * (r_j - r_i) / r

    Parameters:
    -----------
    A : float
        Maximum repulsion force when r = 0.
    rc : float
        Cutoff distance (Angstroms) where the interaction goes to zero.
    pairs : collection of 2-tuples, optional
        Atom symbol pairs that interact, e.g. [('H', 'O'), ('O', 'O')].
        Order within each tuple does not matter: ('H', 'O') == ('O', 'H').
        If None (default), all pairs interact.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, A=25.0, rc=1.0, pairs=None, **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.rc = rc
        # Store as a set of frozensets so ('H', 'O') == ('O', 'H')
        self.pairs = (
            None if pairs is None
            else {frozenset(p) for p in pairs}
        )

    def _pair_mask(self, symbols, i, j):
        """Boolean mask: True where the (i, j) pair is in self.pairs."""
        if self.pairs is None:
            return np.ones(len(i), dtype=bool)
        return np.array(
            [frozenset((symbols[a], symbols[b])) in self.pairs for a, b in zip(i, j)]
        )

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        i, j, d, D = neighbor_list('ijdD', self.atoms, self.rc)

        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))

        if len(d) > 0:
            # Filter to only the specified pairs
            symbols = self.atoms.get_chemical_symbols()
            mask = self._pair_mask(symbols, i, j)
            i, j, d, D = i[mask], j[mask], d[mask], D[mask]

        if len(d) > 0:
            # --- Energy ---
            # V(r) = 0.5 * A * rc * (1 - r/rc)^2
            # The 0.5 in the sum corrects for (i,j)+(j,i) double-counting by neighbor_list
            pair_energies = self.A * self.rc * (1.0 - d / self.rc)**2
            energy = 0.5 * np.sum(pair_energies)

            # --- Forces ---
            # To avoid division by zero when atoms exactly overlap (d=0),
            # we use a safe minimum. If d=0, D is [0,0,0] so the force vector stays [0,0,0].
            safe_d = np.maximum(d, 1e-8)
            pair_forces_magnitudes = self.A * (1.0 - d / self.rc)
            pair_forces = -(pair_forces_magnitudes / safe_d)[:, np.newaxis] * D
            np.add.at(forces, i, pair_forces)

        self.results['energy'] = energy
        self.results['forces'] = forces