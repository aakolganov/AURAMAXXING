"""Write a CP2K QUICKSTEP/GPW geometry-optimization input (.inp + coord.xyz).

PBE-D3BJ, GTH-PBE pseudopotentials with the DZVP-MOLOPT-SR-GTH basis, 500/50 Ry grid,
EPS_SCF 1e-6 Hartree, OT/DIIS, spin-restricted, no wavefunction restart files. The whole
input is built as a nested-section tree so a user can deep-merge overrides/additions from
YAML (config ``dft.cp2k.input``) against any section or keyword.
"""
from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from .common import deep_merge, fixed_indices, fmt_scalar, sort_by_Z

DEFAULT_CP2K_BASIS = "DZVP-MOLOPT-SR-GTH"

# Valence electron count for the GTH-PBE potential name (GTH-PBE-q{N}); falls back to plain
# GTH-PBE for elements not listed.
CP2K_GTH_VALENCE: dict = {"H": 1, "O": 6, "Al": 3, "Si": 4}


def _potential_for(symbol: str) -> str:
    q = CP2K_GTH_VALENCE.get(symbol)
    return f"GTH-PBE-q{q}" if q is not None else "GTH-PBE"


def _fmt_cp2k(value) -> str:
    return fmt_scalar(value, true="TRUE", false="FALSE")   # CP2K booleans: TRUE / FALSE


def _render_section(name: str, body: dict, indent: int) -> str:
    pad = "  " * indent
    param = body.get("_")
    header = f"{pad}&{name}" + (f" {param}" if param is not None else "")
    inner = render_cp2k(body, indent + 1)
    footer = f"{pad}&END {name}"
    return "\n".join(part for part in (header, inner, footer) if part)


def render_cp2k(tree: dict, indent: int = 0) -> str:
    """Render a nested-section dict to CP2K input text. Conventions: key ``"_"`` is the
    section parameter (text after ``&NAME``); a dict value is a subsection; a list of dicts
    is a repeated subsection (e.g. several ``&KIND``); anything else is a ``KEY value`` line."""
    pad = "  " * indent
    lines = []
    for key, val in tree.items():
        if key == "_":
            continue
        if isinstance(val, dict):
            lines.append(_render_section(key, val, indent))
        elif isinstance(val, list) and val and all(isinstance(v, dict) for v in val):
            lines.extend(_render_section(key, item, indent) for item in val)
        else:
            lines.append(f"{pad}{key} {_fmt_cp2k(val)}")
    return "\n".join(lines)


def build_cp2k_tree(atoms: Atoms, project: str, basis_file: str, potential_file: str):
    """Build the default CP2K input tree (and the Z-sorted atoms its coord.xyz/constraints
    refer to) for ``atoms``."""
    sorted_atoms = sort_by_Z(atoms)
    cell = sorted_atoms.cell[:]
    symbols = list(dict.fromkeys(sorted_atoms.get_chemical_symbols()))
    kinds = [{"_": sym, "ELEMENT": sym, "BASIS_SET": DEFAULT_CP2K_BASIS,
              "POTENTIAL": _potential_for(sym)} for sym in symbols]

    tree: dict = {
        "GLOBAL": {"PROJECT": project, "RUN_TYPE": "GEO_OPT", "PRINT_LEVEL": "LOW"},
        "FORCE_EVAL": {
            "METHOD": "QUICKSTEP",
            "DFT": {
                "BASIS_SET_FILE_NAME": basis_file,
                "POTENTIAL_FILE_NAME": potential_file,
                "MGRID": {"CUTOFF": 500, "REL_CUTOFF": 50},
                "QS": {"METHOD": "GPW", "EPS_DEFAULT": "1.0E-10"},
                "SCF": {
                    "MAX_SCF": 50,
                    "EPS_SCF": "1.0E-6",
                    "SCF_GUESS": "ATOMIC",
                    "OT": {"MINIMIZER": "DIIS", "PRECONDITIONER": "FULL_SINGLE_INVERSE"},
                    "PRINT": {"RESTART": {"_": "OFF"}, "RESTART_HISTORY": {"_": "OFF"}},
                },
                "XC": {
                    "XC_FUNCTIONAL": {"_": "PBE"},
                    "VDW_POTENTIAL": {
                        "POTENTIAL_TYPE": "PAIR_POTENTIAL",
                        "PAIR_POTENTIAL": {
                            "TYPE": "DFTD3(BJ)",
                            "PARAMETER_FILE_NAME": "dftd3.dat",
                            "REFERENCE_FUNCTIONAL": "PBE",
                        },
                    },
                },
                "POISSON": {"PERIODIC": "XYZ"},
            },
            "SUBSYS": {
                "CELL": {"A": list(cell[0]), "B": list(cell[1]), "C": list(cell[2]),
                         "PERIODIC": "XYZ"},
                "TOPOLOGY": {"COORD_FILE_NAME": "coord.xyz", "COORD_FILE_FORMAT": "XYZ"},
                "KIND": kinds,
            },
        },
        "MOTION": {"GEO_OPT": {"OPTIMIZER": "BFGS", "MAX_ITER": 200}},
    }

    fixed = sorted(fixed_indices(sorted_atoms))
    if fixed:
        # CP2K atom indices are 1-based and refer to the coord.xyz order (Z-sorted).
        tree["MOTION"]["CONSTRAINT"] = {
            "FIXED_ATOMS": {"LIST": " ".join(str(i + 1) for i in fixed)}
        }
    return tree, sorted_atoms


def write_cp2k_inputs(atoms: Atoms, options: dict | None, out_dir: Path) -> Path:
    """Write ``<project>.inp`` and ``coord.xyz`` into ``out_dir``."""
    options = options or {}
    project = options.get("project", "amorphous_oxide")
    basis_file = options.get("basis_set_file", "BASIS_MOLOPT")
    potential_file = options.get("potential_file", "GTH_POTENTIALS")

    tree, sorted_atoms = build_cp2k_tree(atoms, project, basis_file, potential_file)
    tree = deep_merge(tree, options.get("input"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{project}.inp").write_text(render_cp2k(tree) + "\n")
    ase_write(out_dir / "coord.xyz", sorted_atoms, format="xyz")
    return out_dir
