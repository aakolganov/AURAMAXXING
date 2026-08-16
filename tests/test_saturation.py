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


# --- charge-correction caps must terminate SURFACES, not the bulk -------------------------
# Stage attribution on production slabs: the spatially blind fallback put 67% of its caps
# on interior sites (30% deeper than 4.5 A) -- bulk silanol that real oxides do not have.
# With limits installed, cap-site candidates are partitioned by distance-to-face: surface-
# band sites win; interior is only used when no surface candidate exists.

def _charged_pair_struct(make_struct, with_limits=True):
    from base.limits import make_limit_flat, fix_limits
    # two identical under-coordinated Si (CN 1 each), one at the top face (z=17) and one
    # interior (z=11, with occupied volume BELOW it so the local-surface criterion sees a
    # true interior, not an underside): same CN, so only the surface partition decides
    # which anchors the neutralising -OH cap.
    s = make_struct(["Si", "O", "Si", "O", "O"],
                    [[10, 10, 17.0], [11.62, 10, 17.0], [10, 10, 11.0], [11.62, 10, 11.0],
                     [10, 10, 8.0]],
                    cell=(20.0, 20.0, 24.0))
    if with_limits:
        make_limit_flat(s, z_val=4.0, is_for="bottom")
        make_limit_flat(s, z_val=18.0, is_for="top")
        fix_limits(s.limits)
    return s


def test_charge_cap_prefers_surface_band(make_struct):
    from saturation.new_sat import _add_charge_cap
    s = _charged_pair_struct(make_struct)
    assert _add_charge_cap(s, want_negative=True, bond_lengths=None, num_samples=250)
    g = s.get_graph(force_rebuild=True)
    new_o = len(s) - 2                            # -OH cap: O then H appended
    assert g.has_edge(new_o, 0), "cap must anchor on the SURFACE Si, not the interior one"
    assert not g.has_edge(new_o, 2)


def test_charge_cap_falls_back_to_interior_when_no_surface_candidate(make_struct):
    from base.limits import make_limit_flat, fix_limits
    from saturation.new_sat import _add_charge_cap
    # only an interior candidate exists: the partition must not block progress
    s = make_struct(["Si", "O"], [[10, 10, 11.0], [11.62, 10, 11.0]], cell=(20.0, 20.0, 24.0))
    make_limit_flat(s, z_val=4.0, is_for="bottom")
    make_limit_flat(s, z_val=18.0, is_for="top")
    fix_limits(s.limits)
    assert _add_charge_cap(s, want_negative=True, bond_lengths=None, num_samples=250)
    assert s.get_graph(force_rebuild=True).has_edge(len(s) - 2, 0)


def test_charge_cap_without_limits_is_unpartitioned(make_struct):
    from saturation.new_sat import _add_charge_cap
    # no limits installed (direct API use): behaviour identical to the old min-CN pick
    s = _charged_pair_struct(make_struct, with_limits=False)
    assert _add_charge_cap(s, want_negative=True, bond_lengths=None, num_samples=250)


def test_charge_cap_surface_criterion_survives_envelope_underfill(make_struct):
    from base.limits import make_limit_flat, fix_limits
    from saturation.new_sat import _add_charge_cap, _near_growth_face
    import numpy as np
    # A deposition film can sit far below its growth ceiling. The film's real top atom
    # (z=13) is 9 A under the limit (z=22): an envelope-distance criterion calls it
    # interior and silently disables the partition; the local-atomic-surface criterion
    # must still classify it as surface and win over the buried candidate.
    s = make_struct(["Si", "O", "Si", "O", "O"],
                    [[10, 10, 13.0], [11.62, 10, 13.0], [10, 10, 9.0], [11.62, 10, 9.0],
                     [10, 10, 5.5]],
                    cell=(20.0, 20.0, 26.0))
    make_limit_flat(s, z_val=6.0, is_for="bottom")
    make_limit_flat(s, z_val=22.0, is_for="top")
    fix_limits(s.limits)
    mask = _near_growth_face(s, [0, 2])          # one entry per passed index
    assert mask[0] and not mask[1], "local-surface criterion must be fill-independent"
    assert _add_charge_cap(s, want_negative=True, bond_lengths=None, num_samples=250)
    g = s.get_graph(force_rebuild=True)
    assert g.has_edge(len(s) - 2, 0), "cap must land on the film's real surface"


def test_charge_cap_never_prefers_substrate_interior(make_struct):
    from base.limits import make_limit_flat, fix_limits
    from saturation.new_sat import _near_growth_face
    import numpy as np
    # substrate run: lower_lim sits at the substrate TOP, so an envelope-based bottom
    # band called the whole substrate "surface" and planted caps inside it (seen in the
    # siral-10 check: a cap H at z=4.4 inside a 0-7.4 A substrate). With the local
    # criterion only the substrate's true underside and the film top classify as faces.
    #        z:  1.0 (underside)  4.0 (substrate mid)  8.0 (interface)  12.0 (film top)
    s = make_struct(["Al", "Al", "Al", "Si"],
                    [[10, 10, 1.0], [10, 10, 4.0], [10, 10, 8.0], [10, 10, 12.0]],
                    cell=(20.0, 20.0, 24.0))
    make_limit_flat(s, z_val=7.0, is_for="bottom")    # substrate top = growth floor
    make_limit_flat(s, z_val=18.0, is_for="top")
    fix_limits(s.limits)
    mask = _near_growth_face(s, [0, 1, 2, 3])
    assert bool(mask[0]) is True,  "substrate underside is a real exposed face"
    assert bool(mask[1]) is False, "substrate mid must NEVER be preferred cap territory"
    assert bool(mask[2]) is False, "buried interface is not an exposed face"
    assert bool(mask[3]) is True,  "film top is a face"


# --- saturation must judge UNDER-coordination on hetero-only (ionic) CN -------------------
# The full bond graph counts same-element contacts (relax-born Si-Si / peroxide O-O inside
# the homo cutoffs) as coordination, letting those defects mask themselves and their
# partners from saturation. Hetero-only CN unmasks them; OVER-coordination keeps the
# full-graph basis so break-and-cap can still target the homo-contact carriers.

def test_saturation_unmasks_si_hidden_by_si_si_contact(make_struct):
    from saturation.new_sat import saturate_under_coordinated, hetero_cn
    d = 1.62
    # Si#0: three O + a bare Si#4 at 2.25 A (inside the legacy 2.3 Si-Si graph cutoff).
    # Full-graph CN 4 used to mask it; hetero CN is 3, so it must receive an -OH cap.
    s = make_struct(["Si", "O", "O", "O", "Si"],
                    [[10, 10, 10], [10 + d, 10, 10], [10 - d, 10, 10], [10, 10 + d, 10],
                     [10, 10 - 1.3, 11.84]],   # 2.25 A from Si#0, away from the O's
                    cell=(20.0, 20.0, 22.0))
    g = s.get_graph(force_rebuild=True)
    assert g.has_edge(0, 4), "fixture needs the masking Si-Si contact in the graph"
    assert int(s.get_cn(0)) == 4 and int(hetero_cn(s)[0]) == 3
    n0 = len(s)
    saturate_under_coordinated(s)
    assert hetero_cn(s)[0] == 4, "masked Si must be unmasked and capped to hetero-CN 4"
    assert len(s) > n0


def test_saturation_unmasks_peroxide_oxygen(make_struct):
    from saturation.new_sat import saturate_under_coordinated, hetero_cn
    # O#1 bonded to Si#0 and to O#2 (peroxide at 1.6 A < 1.8 O-O cutoff): full CN 2 used
    # to mask it; hetero CN 1 -> it must receive an H cap. O#2 (hetero CN 1 via its own
    # Si#3) likewise.
    s = make_struct(["Si", "O", "O", "Si"],
                    [[10, 10, 10], [11.62, 10, 10], [12.4, 10, 11.4], [13.2, 10, 12.8]],
                    cell=(20.0, 20.0, 24.0))
    g = s.get_graph(force_rebuild=True)
    assert g.has_edge(1, 2), "fixture needs the O-O bond in the graph"
    assert int(s.get_cn(1)) == 2 and int(hetero_cn(s)[1]) == 1
    saturate_under_coordinated(s)
    h = hetero_cn(s)
    assert h[1] >= 2 and h[2] >= 2, "peroxide oxygens must be unmasked and capped"


def test_over_collection_keeps_full_graph_basis(make_struct):
    from saturation.new_sat import collect_over_or_under_cn_atoms
    # O#1 with two Si plus an O-O contact: full CN 3 (over max 2) but hetero CN 2. The
    # OVER collector must still flag it -- that is how break-and-cap targets homo defects.
    s = make_struct(["Si", "O", "Si", "O"],
                    [[10, 10, 10], [11.62, 10, 10], [13.24, 10, 10], [11.62, 10, 11.6]],
                    cell=(20.0, 20.0, 22.0))
    g = s.get_graph(force_rebuild=True)
    assert g.has_edge(1, 3)
    over = collect_over_or_under_cn_atoms(s, do_under=False)
    assert 1 in [int(i) for i in over.get("O", [])]


# --- detached all-anion fragments: drop + repay charge as -OH, never cap in place ---------
# An MLIP relax can eject a weakly bound O2; blind capping turned it into gas-phase H2O2
# (seen in production). The fragment is dropped and its formal charge repaid on the slab.

def test_detached_o2_dropped_and_repaid_as_hydroxyls(make_struct):
    import numpy as np
    from saturation.new_sat import saturate_under_coordinated
    d = 1.62
    # a small saturable network (Si + 2 O) ... plus a detached O2 far away
    s = make_struct(["Si", "O", "O", "O", "O"],
                    [[10, 10, 10], [10 + d, 10, 10], [10 - d, 10, 10],
                     [4, 4, 16], [4, 4, 17.3]],
                    cell=(20.0, 20.0, 22.0))
    from saturation.new_sat import drop_detached_anion_fragments
    q0 = s.charge()
    removed = drop_detached_anion_fragments(s)
    assert removed == 2
    # the pass itself is ledger-neutral: -4 removed with the O2, -4 repaid as 4 -OH groups
    assert s.charge() == q0, "fragment charge must be repaid exactly (ledger-neutral pass)"
    syms = np.array(s.symbols)
    assert list(syms).count("H") == 4 and list(syms).count("O") == 6   # 2 network + 4 cap O
    pos = s.atoms.get_positions()
    dd = np.sqrt(((pos - np.array([4, 4, 16.5])) ** 2).sum(1))
    assert not (dd < 2.0).any(), "detached O2 must be removed, not capped in place"
    saturate_under_coordinated(s)   # and the full stage still runs cleanly afterwards"


def test_detached_cation_fragment_is_left_alone(make_struct, capsys):
    from saturation.new_sat import drop_detached_anion_fragments
    # a detached SiO cluster contains a cation: must NOT be silently deleted
    s = make_struct(["Si", "O", "O", "Si", "O"],
                    [[10, 10, 10], [11.62, 10, 10], [8.38, 10, 10],
                     [4, 4, 16], [4, 4, 17.77]],
                    cell=(20.0, 20.0, 22.0))
    removed = drop_detached_anion_fragments(s)
    assert removed == 0
    assert "WARNING" in capsys.readouterr().out
    assert len(s) == 5


# --- saturation caps are bystander-aware (terminal placement) -----------------------------

def test_saturation_cap_avoids_bystander_bond(make_struct):
    import numpy as np
    from saturation.new_sat import saturate_under_coordinated
    # dangling O#1 (on Si#0) needs an H cap; bystander O#2 sits 2.1 A away, so part of the
    # H sphere would bond BOTH oxygens. Terminal placement must bond only the anchor.
    s = make_struct(["Si", "O", "O", "Si"],
                    [[10, 10, 10], [11.62, 10, 10], [12.6, 10, 11.7], [13.4, 10, 13.2]],
                    cell=(20.0, 20.0, 24.0))
    saturate_under_coordinated(s)
    syms = np.array(s.symbols)
    pos = s.atoms.get_positions()
    L = s.atoms.cell.lengths()
    for i in np.where(syms == "H")[0]:
        d = pos[np.array(syms) == "O"] - pos[i]
        d -= np.round(d / L) * L
        n_o = int((np.sqrt((d**2).sum(1)) < 1.2).sum())
        assert n_o == 1, "cap H must bond exactly its anchor oxygen"


# --- break-and-cap must never select a cap as break material ------------------------------

def test_break_step_never_breaks_existing_caps(make_struct):
    from saturation.new_sat import (_break_and_cap_step, _set_cap_role, NEG_CAP, OH_H)
    d = 1.62
    # a bridging OH: cap-O#3 bonded to Si#0 AND Si#4, carrying H#5 -> full CN 3 (over max 2),
    # the only over-CN anion in the system. As a cap it must be excluded -> step declines.
    s = make_struct(["Si", "O", "O", "O", "Si", "H"],
                    [[10, 10, 10], [10 + d, 10, 10], [10 - d, 10, 10],
                     [10, 11.3, 11.0], [10, 11.3, 12.9], [10.7, 11.8, 10.8]],
                    cell=(20.0, 20.0, 24.0))
    _set_cap_role(s, 3, NEG_CAP)
    _set_cap_role(s, 5, OH_H)
    assert int(s.get_cn(3)) > 2, "fixture: the cap-O must read over-coordinated"
    out = _break_and_cap_step(s, current_charge=2, bond_lengths=None,
                              num_samples=100, move_alpha=0.5)
    assert out is None, "with only cap material available the break step must decline"
    g = s.get_graph(force_rebuild=True)
    assert g.has_edge(3, 0) or g.has_edge(3, 4), "the OH bridge must remain intact"
