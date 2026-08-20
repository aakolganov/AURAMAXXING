import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

class SoftCoreRepulsion(Calculator):
    """
    Classic soft-core repulsive calculator (as used in DPD conservative forces).
    
    Potential: V(r) = 0.5 * A * rc * (1 - r/rc)^2
    Force:     F(r) = A * (1 - r/rc) * (r_j - r_i) / r
    
    Parameters:
    -----------
    A : float
        Maximum repulsion force when r = 0.
    rc : float
        Cutoff distance (Angstroms) where the interaction goes to zero.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, A=25.0, rc=1.0, **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.rc = rc

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Get neighbor list using the cutoff rc
        # i: index of first atom
        # j: index of second atom
        # d: distance between atoms (scalar r)
        # D: distance vector (r_j - r_i)
        i, j, d, D = neighbor_list('ijdD', self.atoms, self.rc)

        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))

        if len(d) > 0:
            # --- Energy Calculation ---
            # V(r) = 0.5 * A * rc * (1 - r/rc)^2
            pair_energies = 0.5 * self.A * self.rc * (1.0 - d / self.rc)**2
            
            # Divide by 2 because neighbor_list returns both (i, j) and (j, i)
            energy = 0.5 * np.sum(pair_energies) 

            # --- Force Calculation ---
            # To avoid division by zero when finding the unit vector if atoms exactly overlap (d=0)
            # we use a tiny safe value. If d=0, D is [0,0,0], so the resulting force vector remains [0,0,0].
            safe_d = np.maximum(d, 1e-8)
            
            # F_mag = A * (1 - r/rc)
            pair_forces_magnitudes = self.A * (1.0 - d / self.rc)
            
            # Multiply scalar magnitudes by the unit vectors (D / safe_d)
            pair_forces = (pair_forces_magnitudes / safe_d)[:, np.newaxis] * D

            # Accumulate forces on each atom
            np.add.at(forces, i, pair_forces)

        self.results['energy'] = energy
        self.results['forces'] = forces