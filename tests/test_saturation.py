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


def test_correct_charge_terminates_when_unsatisfiable(make_struct):
    from saturation.new_sat import correct_charge
    # a lone Si is charged (+4) but has no anion / tetrahedral site to attach to,
    # so correct_charge cannot neutralise it. It must stop (not loop forever) AND must
    # not silently ship a charged slab -- it raises rather than returning charged.
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert s.charge() != 0
    with pytest.raises(ValueError, match="could not neutralise"):
        correct_charge(s, max_iterations=50)


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
