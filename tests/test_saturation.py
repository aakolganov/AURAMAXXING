"""Tests for the saturation routines."""
import pytest


# --- P2: saturation must place groups at physical bond lengths --------------------
# Regression for hardcoded bond_lengths={"O": 1.2, "H": 0.96}: the O bonded to an
# under-coordinated Si/Al sat at 1.2 A (Si-O is ~1.6 A; El_O_BONDLENGTH = 1.63),
# while the matching constants in default_constants.py went unused.

def test_saturation_uses_physical_bond_lengths(make_struct, monkeypatch, tmp_path):
    # highlight_coordination() writes a file into CWD; keep it out of the repo tree.
    monkeypatch.chdir(tmp_path)

    from saturation.new_sat import saturate_under_coordinated
    from default_constants import El_O_BONDLENGTH, O_H_BONDLENGTH

    # A lone Si is under-coordinated (CN 0 < min_cn 4) and is a cation, so it gets -OH.
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    saturate_under_coordinated(s)

    syms = s.symbols
    assert syms.count("O") == 1
    assert syms.count("H") == 1

    si, o, h = syms.index("Si"), syms.index("O"), syms.index("H")
    d_si_o = s.atoms.get_distance(si, o, mic=True)
    d_o_h = s.atoms.get_distance(o, h, mic=True)

    # The added Si-O bond must be ~1.6 A, not the old 1.2 A.
    assert d_si_o == pytest.approx(El_O_BONDLENGTH, abs=0.05)
    assert d_o_h == pytest.approx(O_H_BONDLENGTH, abs=0.05)


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
