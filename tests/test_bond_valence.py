"""Tests for the bond-valence-deficit anchor ranking in charge correction (Fix A).

Pins the two pathologies the ranking exists to fix:
- the geminal factory: the old lowest-CN tie-break re-selected a just-capped Si for the
  next -OH, producing ~37% geminal silanols (real silica: <~15%);
- the Al compensation gap: an H cap always went to the lowest-CN anion, so the bridging O
  of an AlO4- unit (a real local charge deficit, Pauling sum 1.75 < 2) could never win
  over any dangling O tie and Bronsted Si-O(H)-Al sites formed only by accident.
"""
import numpy as np
import pytest

from base.amorphous_structure import AmorphousStruc_factory
from saturation.new_sat import (bond_valence_deficits, _carries_hydroxyl,
                                _add_charge_cap)


def _struct(symbols, positions, cell=(24.0, 24.0, 24.0), seed=7):
    return AmorphousStruc_factory(symbols=list(symbols),
                                  positions=np.asarray(positions, dtype=float),
                                  cell=list(cell), pbc=True, seed=seed)


def test_deficit_values_on_hand_built_motifs():
    # Si-O-Si bridge, a dangling O, a silanol O, all on one line-free layout.
    #   0 Si  1 O(bridge)  2 Si  3 O(dangling on Si2)  4 O(silanol O on Si0)  5 H
    s = _struct(
        ["Si", "O", "Si", "O", "O", "H"],
        [[10, 10, 10], [11.6, 10, 10], [13.2, 10, 10],
         [13.2, 11.7, 10], [10, 11.7, 10], [10, 12.68, 10]],
    )
    d = bond_valence_deficits(s)
    # In this truncated fragment both Si have het CN 2 and so donate 4/2 = 2 per O.
    # The absolute values are truncation artefacts; what they pin is the FORMULA:
    assert d[0] == pytest.approx(2.0, abs=1e-9)    # Si0: 4 - 2/2 (bridge O) - 2/2 (silanol O)
    assert d[1] == pytest.approx(-2.0, abs=1e-9)   # bridge O: 2 - 2 - 2
    assert d[3] == pytest.approx(0.0, abs=1e-9)    # dangling O: 2 - 2
    assert d[4] == pytest.approx(-1.0, abs=1e-9)   # silanol O: 2 - 2 - 1 (H donates 1/1)
    assert d[5] == pytest.approx(0.0, abs=1e-9)    # H: 1 - 2/2 (its O has het CN 2)
    # and the ordering property the anchor ranking relies on: the dangling O out-ranks
    # every saturated oxygen in the fragment
    assert d[3] > d[4] > d[1]


def test_deficit_ranks_al_bridging_oxygen_above_siosi():
    # Two complete tetrahedra sharing nothing, each cation with 4 oxygens, one Si and one
    # Al; every O also gets a second cation (Si) so all O are 2-coordinated bridges.
    # Build: central Si at A, central Al at B, each with 4 O; each O capped by an outer Si
    # placed opposite, making every O a two-cation bridge with realistic het CNs.
    syms, pos = [], []

    def tetra(center, cat):
        i0 = len(syms)
        syms.append(cat)
        pos.append(center)
        c = np.array(center)
        for v in ([1.6, 0, 0], [-1.6, 0, 0], [0, 1.6, 0], [0, -1.6, 0]):
            syms.append("O")
            pos.append((c + v).tolist())
            syms.append("Si")                       # outer cation completing the bridge
            pos.append((c + np.array(v) * 2.0).tolist())
        return i0

    si0 = tetra([6, 6, 6], "Si")
    al0 = tetra([6, 6, 14], "Al")
    s = _struct(syms, pos, cell=(28.0, 28.0, 28.0))
    d = bond_valence_deficits(s)
    o_on_si = [j for j in s.get_graph()[si0]]
    o_on_al = [j for j in s.get_graph()[al0]]
    # outer Si have het CN 1 and donate 4 to their O; the comparison that matters is
    # BETWEEN the two central units' oxygens: Al(3+, CN4) donates 0.75 vs Si's 1.0, so
    # every O on the Al unit carries a strictly larger deficit than its Si-unit twin.
    assert min(d[j] for j in o_on_al) > max(d[j] for j in o_on_si) + 0.2


def test_charge_cap_prefers_bare_si_over_silanol_bearing_si():
    # Two CN-3 silicons: Si_a bare, Si_b already carrying a silanol (-OH). Old rule tied
    # on CN and picked the lower index (the capped one when it comes first); the new
    # ranking must pick the BARE Si regardless of index order.
    syms, pos = [], []

    def cn3_si(center, with_oh):
        c = np.array(center)
        i0 = len(syms)
        syms.append("Si")
        pos.append(center)
        for v in ([1.6, 0, 0], [-0.8, 1.39, 0], [-0.8, -1.39, 0]):
            syms.append("O")
            pos.append((c + np.array(v)).tolist())
        if with_oh:
            syms.append("O")
            pos.append((c + [0, 0, 1.6]).tolist())
            syms.append("H")
            pos.append((c + [0, 0, 2.58]).tolist())
        return i0

    si_capped = cn3_si([7, 7, 7], True)     # lower index: the old rule's pick
    si_bare = cn3_si([17, 17, 17], False)
    s = _struct(syms, pos, cell=(26.0, 26.0, 26.0))
    assert _carries_hydroxyl(s, si_capped) and not _carries_hydroxyl(s, si_bare)

    n0 = len(s)
    assert _add_charge_cap(s, want_negative=True, bond_lengths=None, num_samples=200)
    new_o = n0                                     # the cap O is appended first
    assert s.atoms[new_o].symbol == "O"
    assert si_bare in list(s.get_graph()[new_o]), \
        "charge -OH must anchor on the bare Si, not re-cap the silanol-bearing one"


def test_charge_cap_targets_al_bridging_oxygen_for_H():
    # No dangling O anywhere; candidates are saturated bridges. The Si-O(H)-Al site must
    # win the H over any Si-O-Si oxygen (Pauling 0.25 vs 0.0 deficit).
    syms, pos = [], []
    # chain:  Si - O - Si - O - Al ; every cation completed to CN>=2 via extra O below
    syms += ["Si", "O", "Si", "O", "Al"]
    pos += [[6, 6, 6], [7.6, 6, 6], [9.2, 6, 6], [10.8, 6, 6], [12.4, 6, 6]]
    for i, c in ((0, [6, 6, 6]), (2, [9.2, 6, 6]), (4, [12.4, 6, 6])):
        for k, v in enumerate(([0, 1.6, 0], [0, -1.6, 0], [0, 0, 1.6])):
            syms.append("O")
            pos.append((np.array(c) + np.array(v)).tolist())
    s = _struct(syms, pos, cell=(26.0, 26.0, 26.0))
    n0 = len(s)
    assert _add_charge_cap(s, want_negative=False, bond_lengths=None, num_samples=200)
    new_h = n0
    assert s.atoms[new_h].symbol == "H"
    anchors = list(s.get_graph()[new_h])
    d = bond_valence_deficits(s)
    # the H must land on an O bonded to the Al (largest-deficit anion in the structure)
    al_oxygens = {j for j in s.get_graph()[4]}
    assert any(a in al_oxygens for a in anchors), \
        f"H anchored at {anchors}, expected one of the Al-bonded oxygens {sorted(al_oxygens)}"
