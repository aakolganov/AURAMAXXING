"""A lightweight test double for ``CalculatorInterface``.

The real calculators (LAMMPS/BKS, MACE) are slow and have heavy dependencies.
``DummyInterface`` satisfies the same contract using ASE's built-in, element-agnostic
Lennard-Jones calculator, so ``optimize``/``anneal`` produce real forces and actually
run — but need neither LAMMPS nor torch. This lets the growth/finalize control flow be
exercised in the *fast* CI tier.

It is intentionally not physically meaningful; tests that use it must assert on
structural invariants (charge, coordination bounds, no steric clashes), never on
specific relaxed geometries.
"""
from pathlib import Path

from ase.calculators.lj import LennardJones

from interfaces.base_interface import CalculatorInterface


class DummyInterface(CalculatorInterface):
    def __init__(self, dump_path: str = "dump_dummy"):
        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)
        # Soft, short-ranged LJ: enough to give finite forces for LBFGS/Langevin
        # without blowing up on the ~1.6 A bonds these structures contain.
        self.calc = LennardJones(sigma=1.5, epsilon=0.01, rc=6.0)
