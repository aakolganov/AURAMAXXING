"""Tests for the saturation routines."""
import pytest


# --- P2: saturation must place groups at physical bond lengths --------------------
# Cap bond lengths come from the shared covalent-radii model (derive_bond_length), so the
# saturation stage places caps at a coherent distance for any element pair -- not the old
# fixed El_O/O_H constants (and not the original 1.2 A Si-O bug that prompted this test).

def test_saturation_uses_physical_bond_lengths(make_struct, monkeypatch, tmp_path):
    # highlight_coordination() writes a file into CWD; keep it out of the repo tree.
    monkeypatch.chdir(tmp_path)

    from saturation.new_sat import saturate_under_coordinated
    from base.element_data import derive_bond_length

    # A lone Si is under-coordinated (CN 0 < min_cn 4) and is a cation, so it gets -OH.
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    saturate_under_coordinated(s)

    syms = s.symbols
    assert syms.count("O") == 1
    assert syms.count("H") == 1

    si, o, h = syms.index("Si"), syms.index("O"), syms.index("H")
    d_si_o = s.atoms.get_distance(si, o, mic=True)
    d_o_h = s.atoms.get_distance(o, h, mic=True)

    # Each cap sits at its covalent-radii-derived (anchor, cap) bond length.
    assert d_si_o == pytest.approx(derive_bond_length("Si", "O"), abs=0.05)
    assert d_o_h == pytest.approx(derive_bond_length("O", "H"), abs=0.05)


# --- C5: correct_charge must not write a debug 'before_opt.vasp' into CWD ----------

def test_correct_charge_no_debug_dump(make_struct, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from saturation.new_sat import correct_charge

    # Si + 2 O is already charge-neutral (+4 - 4 = 0), so correct_charge just sorts.
    s = make_struct(["Si", "O", "O"], [[10, 10, 10], [11.6, 10, 10], [8.4, 10, 10]],
                    cell=(20.0, 20.0, 20.0))
    assert s.charge() == 0
    correct_charge(s)

    assert not (tmp_path / "before_opt.vasp").exists()


# --- Tier A #3: correct_charge must terminate (neutralise, or stop cleanly) -------

def test_correct_charge_noop_on_neutral(make_struct):
    from saturation.new_sat import correct_charge
    s = make_struct(["Si", "O", "O"], [[10, 10, 10], [11.6, 10, 10], [8.4, 10, 10]],
                    cell=(20.0, 20.0, 20.0))
    assert s.charge() == 0
    correct_charge(s)                 # already neutral -> no-op, must return
    assert s.charge() == 0


def test_correct_charge_neutralises_lone_cation_via_fallback(make_struct):
    from saturation.new_sat import correct_charge
    # A lone Si (+4) has no over/under/variable-CN site for the bond-break strategy, so the
    # direct-cap fallback caps it with -OH four times -> Si(OH)4, charge 0. (This used to raise
    # "unsatisfiable"; the fallback now neutralises any single charged cation/anion.)
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert s.charge() == 4
    correct_charge(s, max_iterations=50)
    assert s.charge() == 0
    syms = s.symbols
    assert syms.count("O") == 4 and syms.count("H") == 4   # four -OH caps


# --- isolated atoms must not crash charge-correction's move selection -------------
# Regression for `np.argmax of an empty sequence`: correct_charge feeds under-coordinated
# atoms to select_idx_for_move, which pivots on the chosen atom's furthest neighbour. An
# isolated (CN 0) candidate has no neighbour, so argmax([]) used to crash the whole slab.

def test_select_idx_for_move_skips_isolated_atoms(make_struct):
    from saturation.new_sat import select_idx_for_move

    # every candidate isolated -> None (caller stops gracefully), no crash
    s = make_struct(["Si", "O"], [[2, 2, 2], [10, 10, 10]], cell=(14.0, 14.0, 14.0))
    assert select_idx_for_move(s, [0, 1]) is None

    # a bonded pair (0,1) plus an isolated candidate (2): the isolated one is never chosen
    s2 = make_struct(["Si", "O", "O"], [[5, 5, 5], [6.6, 5, 5], [11, 11, 11]],
                     cell=(14.0, 14.0, 14.0))
    pair = select_idx_for_move(s2, [0, 1, 2])
    assert pair is not None and set(pair) <= {0, 1}


# --- charge correction must not leave (or crash on) atoms its own move orphaned --------
# move_atom shoves an atom a full bond-length off a neighbour to break a bond; if that
# neighbour had no other bond it is orphaned. When the re-cap placement fails, the dangling
# atom must be pruned rather than left in the structure / fed to the next move selection.

def test_correct_charge_prunes_move_orphans(make_struct):
    import numpy as np
    from saturation.new_sat import move_atom, _prune_orphans_from_move

    s = make_struct(["Si", "O", "O", "O"],
                    [[7, 7, 7], [8.6, 7, 7], [7, 8.6, 7], [7, 7, 8.6]], cell=(16.0, 16.0, 16.0))
    cn_before = s.get_cn()
    n_before = len(s)
    assert int(cn_before[1]) == 1                    # O#1 bonded only to the central Si

    move_atom(s, idx_move=0, move_away_from=1, dist_move=3.5, alpha=0.5)   # orphans O#1
    removed = _prune_orphans_from_move(s, cn_before, n_before)

    assert removed == 1                              # the orphaned O is dropped
    s.get_graph(force_rebuild=True)
    assert int(np.sum(np.asarray(s.get_cn()) == 0)) == 0   # nothing left isolated


# --- M7: correct_charge must drive a fixable charged slab to net-zero formal charge -----
# The other correct_charge tests cover only degenerate paths (already neutral, or unsatisfiable
# -> stop). These exercise the convergence loop itself and assert it reaches charge() == 0.

def test_correct_charge_neutralises_negative_slab(make_struct):
    from saturation.new_sat import correct_charge
    # Two Al each bonded to two O: net formal charge 2*(+3) + 4*(-2) = -2. Each under-coordinated
    # Al is capped with an H (+1), so the loop drives the charge to zero.
    s = make_struct(["Al", "Al", "O", "O", "O", "O"],
                    [[5, 5, 5], [5, 5, 12], [6.8, 5, 5], [3.2, 5, 5], [6.8, 5, 12], [3.2, 5, 12]],
                    cell=(16.0, 16.0, 20.0))
    assert s.charge() == -2
    n0 = len(s)
    correct_charge(s, max_iterations=50)
    assert s.charge() == 0          # the loop converged (not a degenerate no-op)
    assert len(s) > n0              # caps were actually added


@pytest.mark.parametrize("seed", range(10))
def test_correct_charge_neutralises_positive_slab(make_struct, seed):
    from saturation.new_sat import correct_charge
    # An over-coordinated O bonded to three Si (a tricluster): net charge 3*(+4) + (-2) = +10.
    # correct_charge adds OH groups (net -1 each) until neutral.
    # Swept over seeds: seed 5 used to *overshoot* zero (a move orphaned a +1 H that the
    # prune deleted on top of the -1 OH cap, so a step jumped -2 to charge -1, then stuck)
    # and silently shipped a charged slab. The revert-on-overshoot loop must reach exactly 0.
    s = make_struct(["O", "Si", "Si", "Si"],
                    [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7], [7, 8.6, 7]],
                    cell=(16.0, 16.0, 16.0), seed=seed)
    assert s.charge() == 10
    correct_charge(s, max_iterations=200)
    assert s.charge() == 0


# --- break-and-cap must anchor each cap on the fragment's correct oxidation sign ---------
# The -OH cap (net -1) must anchor on a CATION and the H cap (net +1) on an ANION. Anchoring on
# the wrong sign still balances the net charge but builds a peroxide (O-O) / hydride (M-H) motif.
# _break_and_cap_step used to cap `idx_furthest` unconditionally, and since the too-positive branch
# feeds over-coordinated anions (high CN), that was usually the anion -> -OH on an anion (peroxide).

@pytest.mark.parametrize("seed", range(6))
def test_correct_charge_caps_anchor_on_correct_sign(make_struct, seed):
    import numpy as np
    from saturation.new_sat import correct_charge, NEG_CAP
    # tricluster (charge +10) drives many break-and-cap -OH additions; every added cap-oxygen
    # (NEG_CAP) must have >=1 cation neighbour (its intended anchor), never zero.
    s = make_struct(["O", "Si", "Si", "Si"],
                    [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7], [7, 8.6, 7]],
                    cell=(16.0, 16.0, 16.0), seed=seed)
    assert s.charge() == 10
    correct_charge(s, max_iterations=200)
    assert s.charge() == 0
    g = s.get_graph(force_rebuild=True)
    syms, ox = s.symbols, s.oxidation
    roles = s.atoms.arrays["cap_role"]
    for i in np.where(roles == NEG_CAP)[0]:
        n_cation = sum(1 for nb in g.neighbors(i) if ox.get(syms[nb], 0) > 0)
        assert n_cation >= 1, f"cap-O #{i} anchored with no cation neighbour (peroxide motif)"


# --- fragment relabel: swap the engine's -OH/H reference caps for 1-valent fragments -------
# The saturation/charge engine always caps with -OH (cations) and H (anions) and tags the
# added atoms (cap_role). relabel_caps swaps those for the run's fragments, charge-preserving
# because every fragment is 1-valent (OH and F are both -1; H and Na are both +1).

@pytest.fixture
def make_struct_caps():
    """Build a seeded struct whose coordination tables also carry F/Na as terminal (CN 1) cap
    elements, so relabel targets have distance/cutoff rows and an oxidation."""
    import numpy as np
    from base.amorphous_structure import AmorphousStruc_factory
    from base.config import CoordinationConfig
    from base.element_data import build_element_tables

    def _make(symbols, positions, cell=(16.0, 16.0, 16.0), seed=0, caps=("F", "Na")):
        elems = sorted(set(symbols) | {"O", "H"} | set(caps))
        t = build_element_tables(elems, max_cn={e: 1 for e in caps}, min_cn={e: 1 for e in caps})
        cfg = CoordinationConfig(max_cn=t["max_cn"], min_cn=t["min_cn"], cut_offs=t["cut_offs"],
                                 sample_dist=t["sample_dist"], d_min_max=t["d_min_max"],
                                 oxidation=t["oxidation"])
        return AmorphousStruc_factory(symbols=list(symbols),
                                      positions=np.asarray(positions, dtype=float),
                                      cell=list(cell), pbc=True, seed=seed, config=cfg)
    return _make


def test_relabel_anion_cap_oh_to_f(make_struct_caps):
    from saturation.new_sat import saturate_under_coordinated, correct_charge, relabel_caps
    s = make_struct_caps(["O", "Si", "Si", "Si"],
                         [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7], [7, 8.6, 7]], seed=5)
    saturate_under_coordinated(s)
    correct_charge(s, max_iterations=200)
    n_oh = s.symbols.count("O") - 1          # OH oxygens added on top of the one network O
    relabel_caps(s, negative_fragment=("F",), positive_fragment=("H",))
    assert s.charge() == 0                    # OH(-1) -> F(-1) preserves neutrality
    assert s.symbols.count("F") >= 1          # -OH groups became F caps
    # cap-O conservation: each of the n_oh cap oxygens is either retyped to F or (if the
    # unrelaxed cluster left it bridging >1 cation) kept as oxide; plus the one network O.
    assert s.symbols.count("F") + s.symbols.count("O") == n_oh + 1


def test_relabel_cation_cap_h_to_na(make_struct_caps):
    from saturation.new_sat import saturate_under_coordinated, correct_charge, relabel_caps
    from saturation.new_sat import POS_CAP
    import numpy as np
    s = make_struct_caps(["Si", "O", "O", "O"],
                         [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7], [7, 8.6, 7]], seed=3)
    saturate_under_coordinated(s)
    correct_charge(s, max_iterations=200)
    n_pos = int(np.sum(np.asarray(s.atoms.arrays["cap_role"]) == POS_CAP))
    relabel_caps(s, negative_fragment=("O", "H"), positive_fragment=("Na",))
    assert s.charge() == 0                    # H(+1) -> Na(+1) preserves neutrality
    assert s.symbols.count("Na") == n_pos     # every standalone H cap became Na


def test_relabel_default_is_noop(make_struct_caps):
    from saturation.new_sat import saturate_under_coordinated, correct_charge, relabel_caps
    s = make_struct_caps(["O", "Si", "Si", "Si"],
                         [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7], [7, 8.6, 7]], seed=1)
    saturate_under_coordinated(s)
    correct_charge(s, max_iterations=200)
    before = sorted(s.symbols)
    relabel_caps(s, negative_fragment=("O", "H"), positive_fragment=("H",))
    assert sorted(s.symbols) == before        # default OH/H -> no substitution


def test_relabel_neutral_both_fragments(make_struct_caps):
    from saturation.new_sat import saturate_under_coordinated, correct_charge, relabel_caps
    s = make_struct_caps(["O", "Si", "Si", "Si"],
                         [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7], [7, 8.6, 7]], seed=7)
    saturate_under_coordinated(s)
    correct_charge(s, max_iterations=200)
    relabel_caps(s, negative_fragment=("F",), positive_fragment=("Na",))
    assert s.charge() == 0                     # both swaps are charge-preserving (mode a)
    # mode a caps both sublattices: F on cations, Na on anions (this slab has cation caps)
    assert s.symbols.count("F") >= 1


def test_relabel_without_caps_is_noop(make_struct_caps):
    # No saturation was run, so there is no cap_role array: relabel must be a safe no-op.
    from saturation.new_sat import relabel_caps
    s = make_struct_caps(["Si", "O", "O"], [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7]], seed=0)
    before = sorted(s.symbols)
    relabel_caps(s, negative_fragment=("F",), positive_fragment=("Na",))
    assert sorted(s.symbols) == before


def test_relabel_preserves_a_nonzero_charge(make_struct_caps):
    # relabel only guards that it does not *break* neutrality; on an already-charged slab
    # (no charge-correction run) it must still substitute and keep the same net charge.
    from saturation.new_sat import saturate_under_coordinated, relabel_caps
    s = make_struct_caps(["Si"], [[8, 8, 8]], seed=0)
    saturate_under_coordinated(s)              # lone Si -> Si-O-H, net charge +4-2+1 = +3
    assert s.charge() == 3
    relabel_caps(s, negative_fragment=("F",), positive_fragment=("H",))
    assert s.charge() == 3                      # OH(-1) -> F(-1): charge unchanged, no raise
    assert s.symbols.count("F") == 1 and s.symbols.count("O") == 0


def test_relabel_survives_split_oh_pair_no_crash(make_struct_caps):
    # A1 regression: correct_charge's orphan-prune can delete a NEG_CAP O or its OH_H
    # independently, so the NEG_CAP/OH_H counts can diverge. relabel must stay charge-balanced
    # (pair one drop per retype) and NOT raise. Build two clean Si-O-H caps, then delete one H
    # to force an imbalance (2 NEG_CAP O, 1 OH_H).
    import numpy as np
    from saturation.new_sat import relabel_caps, NEG_CAP, OH_H, CAP_NONE
    s = make_struct_caps(["Si", "O", "H", "Si", "O", "H"],
                         [[3, 3, 3], [4.6, 3, 3], [5.5, 3, 3],
                          [10, 10, 10], [11.6, 10, 10], [12.5, 10, 10]], seed=0)
    s.atoms.set_array("cap_role",
                      np.array([CAP_NONE, NEG_CAP, OH_H, CAP_NONE, NEG_CAP, OH_H], dtype=int))
    s.remove_atom([5])                       # drop one OH_H -> 2 NEG_CAP O but only 1 OH_H
    c0 = s.charge()
    relabel_caps(s, negative_fragment=("F",), positive_fragment=("H",))  # must not raise
    assert s.charge() == c0                  # charge preserved despite the count imbalance
    assert s.symbols.count("F") == 1         # only one O could be paired with an OH_H to drop


def test_relabel_skips_bridging_cap_oxygen(make_struct_caps):
    # A2 regression: a cap-O the relax pulled into a 2-cation bridge is not a clean terminal
    # cap; retyping it to a 1-valent halide would bond F to two cations. It must be left as is.
    import numpy as np
    from saturation.new_sat import relabel_caps, NEG_CAP, OH_H, CAP_NONE
    s = make_struct_caps(["Si", "Si", "O", "H"],
                         [[3, 3, 3], [6.2, 3, 3], [4.6, 3, 3], [4.6, 3.9, 3]], seed=0)
    s.atoms.set_array("cap_role", np.array([CAP_NONE, CAP_NONE, NEG_CAP, OH_H], dtype=int))
    s.get_graph(force_rebuild=True)
    assert s.get_cn(2) == 3                   # the cap-O bridges both Si and holds its H
    c0 = s.charge()
    relabel_caps(s, negative_fragment=("F",), positive_fragment=("H",))
    assert s.charge() == c0
    assert s.symbols.count("F") == 0          # bridging cap-O left as oxide, not retyped to F
    assert s.symbols.count("O") == 1


def test_saturation_is_data_driven_for_non_silica_cation(make_struct_caps):
    # A lone Ga (oxidation +3, a curated non-Si/Al cation) is under-coordinated and must be
    # capped with -OH from its oxidation sign alone -- proving saturation isn't Si/Al-hardcoded.
    from saturation.new_sat import saturate_under_coordinated, NEG_CAP
    s = make_struct_caps(["Ga"], [[8, 8, 8]], seed=0)
    saturate_under_coordinated(s)
    syms = s.symbols
    assert syms.count("O") == 1 and syms.count("H") == 1
    o = syms.index("O")
    assert s.atoms.arrays["cap_role"][o] == NEG_CAP


def test_cap_bond_length_uses_struct_bond_factor(make_struct):
    # D1: caps are placed at the run's bond_factor scale, not always 1.0x.
    from saturation.new_sat import saturate_under_coordinated
    from base.element_data import derive_bond_length
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    s.bond_factor = 1.3
    saturate_under_coordinated(s)
    syms = s.symbols
    d_si_o = s.atoms.get_distance(syms.index("Si"), syms.index("O"), mic=True)
    assert d_si_o == pytest.approx(derive_bond_length("Si", "O", bond_factor=1.3), abs=0.05)


def test_derive_bond_length_matches_covalent_radii():
    from base.element_data import derive_bond_length
    from ase.data import atomic_numbers, covalent_radii
    expected = float(covalent_radii[atomic_numbers["Si"]] + covalent_radii[atomic_numbers["O"]])
    assert derive_bond_length("Si", "O") == pytest.approx(expected, abs=1e-9)
    # symmetric and scaled by bond_factor
    assert derive_bond_length("O", "H") == pytest.approx(derive_bond_length("H", "O"), abs=1e-12)
    assert derive_bond_length("Si", "O", bond_factor=0.5) == pytest.approx(expected * 0.5, abs=1e-9)


# --- the break-and-cap "bond break" must actually sever the bond --------------------------
# _break_and_cap_step moves the defect atom to d_min_max[..][1] + 0.2, past the pair's graph
# cutoff in both bond-table families (legacy: window max 1.92 vs cutoff 2.0; derived:
# d_min_max[1] == the cutoff itself). It used to move to [0] + 0.2 -- the collision floor,
# INSIDE the bonding cutoff -- so the "broken" bond survived every committed step: the charge
# still reached zero (via the cap), but the original CN defect was never repurposed.

@pytest.mark.parametrize("seed", range(10))
def test_break_and_cap_resolves_the_over_coordination(make_struct, seed):
    from saturation.new_sat import correct_charge, collect_over_or_under_cn_atoms
    # tricluster: an O bonded to three Si (CN 3 > max 2). Only a real bond break can lower an
    # anion's CN (caps only ever add bonds), so no over-coordinated anion may survive.
    s = make_struct(["O", "Si", "Si", "Si"],
                    [[7, 7, 7], [8.6, 7, 7], [5.4, 7, 7], [7, 8.6, 7]],
                    cell=(16.0, 16.0, 16.0), seed=seed)
    assert s.get_cn(0) == 3
    correct_charge(s, max_iterations=200)
    assert s.charge() == 0
    over = collect_over_or_under_cn_atoms(s, do_under=False)
    over_anions = [i for k, v in over.items() if s.oxidation.get(k, 0) < 0 for i in v]
    assert not over_anions, "over-coordinated O survived: the bond break never broke the bond"


# --- the direct-cap fallback must never anchor a cap on a monovalent cap atom -------------

def test_charge_cap_fallback_skips_hydrogen_anchors(make_struct):
    import numpy as np
    from saturation.new_sat import _add_charge_cap
    # Si(OH)4: every atom fully coordinated. The hydrogens (CN 1, oxidation +1) used to win
    # the lowest-CN tie-break, so the new cap-O bonded an existing hydroxyl H -- a divalent H
    # bridging two oxygens. Only the network cation (Si) is an eligible anchor.
    d_si_o, d_o_h = 1.62, 0.96
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0)
    c = np.array([8.0, 8.0, 8.0])
    symbols = ["Si"] + ["O"] * 4 + ["H"] * 4
    positions = ([c] + [c + d * d_si_o for d in dirs]
                 + [c + d * (d_si_o + d_o_h) for d in dirs])
    s = make_struct(symbols, positions, cell=(16.0, 16.0, 16.0))
    assert list(s.get_cn()) == [4, 2, 2, 2, 2, 1, 1, 1, 1]   # fully coordinated
    assert _add_charge_cap(s, want_negative=True, bond_lengths=None, num_samples=250)
    cn = s.get_cn()
    h_idx = [i for i, sym in enumerate(s.symbols) if sym == "H"]
    assert all(cn[i] <= 1 for i in h_idx), "cap anchored on a hydroxyl H (divalent H motif)"
    assert cn[0] == 5, "the -OH cap must anchor on the network cation (Si)"


# --- seeded saturation must be reproducible across interpreter processes ------------------
# collect_over_or_under_cn_atoms decides the element capping order (and the break-step
# candidate order); it used to iterate set(symbols), whose order rides on PYTHONHASHSEED --
# the same config+seed produced different slabs in different processes (e.g. spawn workers).

def test_saturation_reproducible_across_hash_seeds(tmp_path):
    import hashlib
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    script = tmp_path / "run_sat_fingerprint.py"
    script.write_text(
        "import hashlib\n"
        "import numpy as np\n"
        "from base.amorphous_structure import AmorphousStruc_factory\n"
        "from saturation.new_sat import saturate_under_coordinated, correct_charge\n"
        "# a Si tricluster defect + an under-coordinated Al2O4 cluster: three element types\n"
        "# with both under- and over-coordination, so element iteration order matters\n"
        "s = AmorphousStruc_factory(\n"
        "    symbols=['O', 'Si', 'Si', 'Si', 'Al', 'Al', 'O', 'O', 'O', 'O'],\n"
        "    positions=np.array([[7., 7., 7.], [8.6, 7., 7.], [5.4, 7., 7.], [7., 8.6, 7.],\n"
        "                        [15., 15., 5.], [15., 15., 12.], [16.8, 15., 5.],\n"
        "                        [13.2, 15., 5.], [16.8, 15., 12.], [13.2, 15., 12.]]),\n"
        "    cell=[22.0, 22.0, 22.0], pbc=True, seed=7)\n"
        "saturate_under_coordinated(s, num_samples=50)\n"
        "correct_charge(s, max_iterations=100, num_samples=50)\n"
        "fp = hashlib.md5(''.join(s.symbols).encode()\n"
        "                 + np.round(s.atoms.get_positions(), 8).tobytes()).hexdigest()\n"
        "print(fp)\n"
    )
    fingerprints = set()
    for hash_seed in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(repo_root))
        out = subprocess.run([sys.executable, str(script)], env=env, cwd=repo_root,
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        fingerprints.add(out.stdout.strip())
    assert len(fingerprints) == 1, f"structure depends on PYTHONHASHSEED: {fingerprints}"
