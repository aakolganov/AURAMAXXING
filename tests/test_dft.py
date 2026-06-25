"""Tests for the DFT-refinement input generator (dft/, pipeline step 3).

Pure file-writing: no VASP/CP2K binary and no MLIP calculator are needed. Structures come
from the seeded ``make_struct`` fixture; the end-to-end runner test uses the LJ dummy backend.
"""
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from dft.api import write_dft_inputs
from dft.common import deep_merge
from dft.cp2k import write_cp2k_inputs
from dft.vasp import write_vasp_inputs
from runner.config import load_config
from runner.runner import run_from_config


# Symbols span H, O, Al, Si so the ascending-Z ordering (H, O, Al, Si) is exercised.
_SYMBOLS = ["Si", "O", "O", "Al", "H"]
_POSITIONS = [[5, 5, 5], [6.6, 5, 5], [5, 6.6, 5], [8, 8, 8], [9, 8, 8]]


def _parse_incar(text: str) -> dict:
    tags = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, val = line.split("=", 1)
            tags[key.strip()] = val.strip()
    return tags


# --- deep_merge --------------------------------------------------------------------

def test_deep_merge_overrides_adds_and_deletes():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    out = deep_merge(base, {"a": 2, "b": {"y": None, "z": 3}, "c": 4})
    assert out == {"a": 2, "b": {"x": 1, "z": 3}, "c": 4}
    # base is not mutated
    assert base == {"a": 1, "b": {"x": 1, "y": 2}}


# --- VASP INCAR --------------------------------------------------------------------

def test_incar_has_required_tags(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    write_vasp_inputs(s.atoms, None, tmp_path / "vasp")
    tags = _parse_incar((tmp_path / "vasp" / "INCAR").read_text())
    assert tags["PREC"] == "Normal"
    assert tags["ENCUT"] == "500"
    assert tags["ALGO"] == "Fast"
    assert tags["EDIFF"] == "1e-05"
    assert tags["EDIFFG"] == "-0.03"
    assert tags["IBRION"] == "2" and tags["POTIM"] == "0.5" and tags["ISIF"] == "2"
    assert tags["ISMEAR"] == "0" and tags["SIGMA"] == "0.05"
    assert tags["ISPIN"] == "1"
    assert tags["IVDW"] == "12"
    assert tags["GGA"] == "PE"
    assert tags["NWRITE"] == "1"
    assert tags["LCHARG"] == ".FALSE." and tags["LWAVE"] == ".FALSE." and tags["LVTOT"] == ".FALSE."


def test_incar_user_override_merges(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    opts = {"incar": {"ENCUT": 600, "ISYM": None, "MY_TAG": 7}}
    write_vasp_inputs(s.atoms, opts, tmp_path / "vasp")
    tags = _parse_incar((tmp_path / "vasp" / "INCAR").read_text())
    assert tags["ENCUT"] == "600"      # overridden
    assert "ISYM" not in tags          # null drops it
    assert tags["MY_TAG"] == "7"       # arbitrary new tag added


# --- VASP KPOINTS ------------------------------------------------------------------

def test_kpoints_gamma_default_and_monkhorst_override(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    write_vasp_inputs(s.atoms, None, tmp_path / "g")
    assert "Gamma" in (tmp_path / "g" / "KPOINTS").read_text()

    write_vasp_inputs(s.atoms, {"kpoints": {"mode": "monkhorst", "grid": [2, 2, 1]}},
                      tmp_path / "m")
    km = (tmp_path / "m" / "KPOINTS").read_text()
    assert "Monkhorst" in km and "2 2 1" in km


# --- VASP POSCAR: ordering + selective dynamics ------------------------------------

def test_poscar_orders_by_atomic_number(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    write_vasp_inputs(s.atoms, None, tmp_path / "vasp")
    atoms = read(str(tmp_path / "vasp" / "POSCAR"), format="vasp")
    assert atoms.get_chemical_symbols() == ["H", "O", "O", "Al", "Si"]
    assert np.all(np.diff(atoms.numbers) >= 0)              # ascending Z
    # POTCAR.order records the same block order for the user's POTCAR
    assert (tmp_path / "vasp" / "POTCAR.order").read_text().strip().endswith("H O Al Si")


def test_poscar_preserves_and_remaps_selective_dynamics(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    s.atoms.set_constraint(FixAtoms(indices=[0]))           # freeze the Si (orig index 0)
    write_vasp_inputs(s.atoms, None, tmp_path / "vasp")
    atoms = read(str(tmp_path / "vasp" / "POSCAR"), format="vasp")
    fixed = sorted(int(i) for c in atoms.constraints for i in c.get_indices())
    assert len(fixed) == 1
    # after Z-sort the Si moved to the last index; it must still be the fixed atom
    assert atoms.get_chemical_symbols()[fixed[0]] == "Si"


# --- CP2K input --------------------------------------------------------------------

def test_cp2k_input_has_required_blocks(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    write_cp2k_inputs(s.atoms, {"project": "amorph"}, tmp_path / "cp2k")
    text = (tmp_path / "cp2k" / "amorph.inp").read_text()
    for needle in ["RUN_TYPE GEO_OPT", "METHOD GPW", "CUTOFF 500", "REL_CUTOFF 50",
                   "EPS_SCF 1.0E-6", "&XC_FUNCTIONAL PBE", "DFTD3(BJ)",
                   "DZVP-MOLOPT-SR-GTH", "GTH-PBE", "&RESTART OFF", "&CELL", "&KIND"]:
        assert needle in text, f"missing {needle!r}"
    # one &KIND per distinct element (H, O, Al, Si)
    assert text.count("&KIND ") == 4
    assert (tmp_path / "cp2k" / "coord.xyz").exists()


def test_cp2k_fixed_atoms_listed_one_based(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    s.atoms.set_constraint(FixAtoms(indices=[0]))           # Si -> sorted index 4 -> 1-based 5
    write_cp2k_inputs(s.atoms, {"project": "amorph"}, tmp_path / "cp2k")
    text = (tmp_path / "cp2k" / "amorph.inp").read_text()
    assert "&FIXED_ATOMS" in text
    assert "LIST 5" in text


def test_cp2k_deep_merge_override(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    opts = {"project": "amorph", "input": {"FORCE_EVAL": {"DFT": {"MGRID": {"CUTOFF": 600}}}}}
    write_cp2k_inputs(s.atoms, opts, tmp_path / "cp2k")
    text = (tmp_path / "cp2k" / "amorph.inp").read_text()
    assert "CUTOFF 600" in text
    assert "CUTOFF 500" not in text
    assert "REL_CUTOFF 50" in text                          # untouched sibling


# --- dispatch ----------------------------------------------------------------------

def test_write_dft_inputs_writes_selected_packages(make_struct, tmp_path):
    s = make_struct(_SYMBOLS, _POSITIONS)
    # api is duck-typed: anything exposing packages/vasp/cp2k works (the runner passes DFTSpec).
    spec = SimpleNamespace(packages=["cp2k"], vasp={}, cp2k={"project": "p"})
    write_dft_inputs(s.atoms, spec, tmp_path / "dft")
    assert (tmp_path / "dft" / "cp2k" / "p.inp").exists()
    assert not (tmp_path / "dft" / "vasp").exists()         # vasp not requested


# --- end-to-end through the runner -------------------------------------------------

def _blank_cfg(tmp_path, dft):
    return load_config({
        "structure": {"cell": [16.0, 16.0, 30.0]},
        "composition": {"target_ratios": {"Si": 1, "O": 2}, "target_number_atoms": 45},
        "limits": {"bottom": {"type": "flat", "z": 10.0},
                   "top": {"type": "fourier", "z_av": 18.0, "alpha": 1.0}, "fix": "bottom"},
        "calculators": {"growth": {"type": "lammps"}},
        "run": {"output_dir": str(tmp_path / "out"), "seeds": [0]},
        "dft": dft,
    })


def test_run_from_config_writes_dft_inputs(tmp_path, monkeypatch):
    from tests.dummy_interface import DummyInterface
    monkeypatch.setattr("runner.runner.build_calculator",
                        lambda spec: DummyInterface(dump_path=str(tmp_path / "dump")))
    cfg = _blank_cfg(tmp_path, {"enabled": True, "packages": ["vasp", "cp2k"],
                                "cp2k": {"project": "test"}})
    run_from_config(cfg)
    dft_dir = tmp_path / "out" / "dft"
    assert (dft_dir / "vasp" / "INCAR").exists()
    assert (dft_dir / "vasp" / "POSCAR").exists()
    assert (dft_dir / "cp2k" / "test.inp").exists()


# --- standalone CLI ----------------------------------------------------------------

def test_cli_generates_inputs_for_directory(tmp_path):
    from dft import cli
    atoms = Atoms("SiO2", positions=[[5, 5, 5], [6.6, 5, 5], [5, 6.6, 5]],
                  cell=[12.0, 12.0, 12.0], pbc=True)
    write(str(tmp_path / "structure.vasp"), atoms, format="vasp")
    rc = cli.main([str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "dft" / "vasp" / "INCAR").exists()
    assert (tmp_path / "dft" / "cp2k" / "amorphous_oxide.inp").exists()
