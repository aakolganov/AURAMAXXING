"""Tests for the structure-modification rules."""


# --- C1: MinimumDistanceRule must push violating atoms APART -----------------------
# Regression for the sign bug: get_distance(a, b, vector=True) returns r_b - r_a
# (points a -> b), and the rule did `positions[a] += that`, moving a *toward* b and
# making the violation worse.

def test_mindistance_pushes_atoms_apart(make_struct):
    from rules import MinimumDistanceRule

    # Two Si only 0.5 A apart, well inside the 1.5 A minimum.
    s = make_struct(["Si", "Si"], [[10.0, 10, 10], [10.5, 10, 10]], cell=(20.0, 20.0, 20.0))
    d_before = s.atoms.get_distance(0, 1, mic=True)

    rule = MinimumDistanceRule(min_dist=1.5, step_size=0.1)
    changed = rule.apply(s.atoms, s.get_graph())

    d_after = s.atoms.get_distance(0, 1, mic=True)
    assert changed is True
    assert d_after > d_before  # distance must increase, not shrink
