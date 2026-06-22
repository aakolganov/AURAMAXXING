"""Tests for anchor/element selection during growth."""
from helpers.atom_picker import choose_atom_idx_to_attach_to, pick_next_atom_type


def test_choose_attach_picks_complementary_element(make_struct):
    s = make_struct(["Si", "O"], [[10, 10, 10], [11.6, 10, 10]], cell=(20.0, 20.0, 20.0))
    # adding O (anion) -> attach to a cation
    assert s.symbols[choose_atom_idx_to_attach_to(s, "O", weight_z=False)] in ("Si", "Al", "H")
    # adding Si (cation) -> attach to an anion
    assert s.symbols[choose_atom_idx_to_attach_to(s, "Si", weight_z=False)] == "O"


def test_choose_attach_excludes_indices(make_struct):
    s = make_struct(["Si", "Si"], [[5, 5, 5], [10, 10, 10]], cell=(20.0, 20.0, 20.0))
    for _ in range(10):
        assert choose_atom_idx_to_attach_to(s, "O", weight_z=False, exclude_indices=[0]) == 1


def test_choose_attach_skips_saturated(make_struct):
    # O0 is bonded to two Si (CN 2 = saturated); O3 is isolated (CN 0).
    # Adding Si must attach to the unsaturated O3, never the saturated O0.
    s = make_struct(["O", "Si", "Si", "O"],
                    [[10, 10, 10], [8.4, 10, 10], [11.6, 10, 10], [15, 15, 15]],
                    cell=(20.0, 20.0, 20.0))
    assert s.get_cn(0) == 2 and s.get_cn(3) == 0
    for _ in range(10):
        assert choose_atom_idx_to_attach_to(s, "Si", weight_z=False) == 3


def test_pick_next_atom_type_follows_deficit(make_struct):
    # all-Si structure, target 1 Si : 2 O -> O is most under-represented
    s = make_struct(["Si", "Si"], [[5, 5, 5], [10, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert pick_next_atom_type(s, {"Si": 1, "O": 2}) == "O"
