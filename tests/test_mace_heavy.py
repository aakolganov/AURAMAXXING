"""Heavy tier: a real MACE relaxation on CPU using a downloaded foundation model.

Marked ``heavy`` so it only runs in the dedicated CI tier (needs torch + mace-torch +
network to fetch the model) — it is collected (so `pytest -m heavy` no longer exits 5)
and skipped where those are unavailable. Uses the MIT-licensed MACE-MP-0a "small" model,
which mace_mp downloads itself; nothing is vendored into the repo.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.heavy

# MIT-licensed MACE-MP-0a small. mace_mp resolves the "small" keyword and downloads it.
# (Confirm the resolved model's release LICENSE if you pin a different variant/URL.)
MODEL = "small"


def test_mace_mp_small_finalize_cpu(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    from interfaces.MACE_interface import MACEInterface
    from base.amorphous_structure import AmorphousStruc_factory

    calc = MACEInterface(mace_model_path=MODEL, device="cpu", dump_path=str(tmp_path / "dump"))

    s = AmorphousStruc_factory(
        symbols=["Si", "O", "O"],
        positions=[[5, 5, 5], [6.6, 5, 5], [3.4, 5, 5]],
        cell=[12.0, 12.0, 12.0], pbc=True, seed=0,
    )
    s.atoms.calc = calc.calc
    assert np.isfinite(s.atoms.get_potential_energy())          # single point works

    calc.optimize(s.atoms, fmax=1.0, max_steps=2)               # a couple of LBFGS steps
    assert np.all(np.isfinite(s.atoms.get_positions()))
