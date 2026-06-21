"""Smoke tests: the package imports and basic structure bookkeeping works.

Cheap, but exactly the class of failure (PEP-604 on 3.9, missing scipy/networkx,
broken module-level imports) that previously could only be caught by running the
whole pipeline.
"""
import numpy as np


def test_core_modules_import():
    import default_constants  # noqa: F401
    from base.initialize import initialize_structure_blank  # noqa: F401
    from growth.new_growth import grow_structure, finalize_structure  # noqa: F401
    from saturation.new_sat import saturate_under_coordinated, correct_charge  # noqa: F401
    from rules import (  # noqa: F401
        PeriodicStructureModifier,
        AvoidMotifSwapRule,
        MinimumDistanceRule,
    )
    from interfaces.base_interface import CalculatorInterface  # noqa: F401
    from interfaces.LAMMPS_Interface import LMPInterface  # noqa: F401


def test_dummy_interface_satisfies_contract():
    from interfaces.base_interface import CalculatorInterface
    from tests.dummy_interface import DummyInterface

    calc = DummyInterface(dump_path="dump_dummy_test")
    assert isinstance(calc, CalculatorInterface)
    assert hasattr(calc, "optimize") and hasattr(calc, "anneal")


def test_basic_coordination_and_charge(make_struct):
    # Si-O at ~1.6 A: each is singly coordinated; net formal charge = +4 + (-2) = +2.
    s = make_struct(["Si", "O"], [[10, 10, 10], [11.6, 10, 10]])
    cn = s.get_cn()
    assert np.array_equal(cn, np.array([1, 1]))
    assert s.charge() == 2
