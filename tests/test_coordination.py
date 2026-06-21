"""Tests for coordination-number classification used by saturation/charge logic."""


# --- P4: over-coordination must be judged against max_cn, not min_cn ---------------
# collect_over_or_under_cn_atoms iterated min_cn and used (cn > min_cn) for the
# "over" case, so an element with min_cn < max_cn (Al: 3..4) flagged a perfectly
# coordinated CN=4 Al as over-coordinated.

def test_over_coordination_uses_max_cn(make_struct):
    from saturation.new_sat import collect_over_or_under_cn_atoms

    # Al with exactly 4 O neighbours (~1.8 A, inside the 2.1 A Al-O cutoff):
    # CN(Al) = 4 = max_cn[Al], so it must NOT count as over-coordinated.
    s = make_struct(
        ["Al", "O", "O", "O", "O"],
        [[10, 10, 10], [11.8, 10, 10], [8.2, 10, 10], [10, 11.8, 10], [10, 8.2, 10]],
        cell=(20.0, 20.0, 20.0),
    )
    assert s.get_cn(0) == 4  # sanity: the Al is 4-coordinate

    over = collect_over_or_under_cn_atoms(s, do_under=False)
    assert 0 not in over.get("Al", [])  # 4-coordinate Al is not over-coordinated


def test_genuinely_over_coordinated_is_flagged(make_struct):
    from saturation.new_sat import collect_over_or_under_cn_atoms

    # Al with 5 O neighbours -> CN 5 > max_cn 4 -> genuinely over-coordinated.
    s = make_struct(
        ["Al", "O", "O", "O", "O", "O"],
        [
            [10, 10, 10],
            [11.8, 10, 10], [8.2, 10, 10],
            [10, 11.8, 10], [10, 8.2, 10],
            [10, 10, 11.8],
        ],
        cell=(20.0, 20.0, 20.0),
    )
    assert s.get_cn(0) == 5
    over = collect_over_or_under_cn_atoms(s, do_under=False)
    assert 0 in over.get("Al", [])
