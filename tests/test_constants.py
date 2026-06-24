"""Consistency checks on the charge constants."""
import pytest


# --- P3: default_charges['Al'] must match bks_charges and keep Al2O3 neutral -------
# default_charges had Al=1.4 while bks_charges (used by LAMMPS) has Al=1.8, which is
# the charge-neutral value for Al2O3.

def test_default_charges_al_matches_bks():
    from default_constants import default_charges, bks_charges
    assert default_charges["Al"] == bks_charges["Al"]


def test_default_charges_give_neutral_al2o3():
    from default_constants import default_charges
    # 2 Al + 3 O must sum to zero formal charge
    assert 2 * default_charges["Al"] + 3 * default_charges["O"] == pytest.approx(0.0)


# --- C5: every bond-length sampling distribution must have non-negative support ----
# The O-O burr12 had loc=-0.226, admitting negative distances.

def test_sample_dist_supports_are_nonnegative():
    from default_constants import sample_dist
    for anchor, rs in sample_dist.items():
        for target in list(rs.keys()):
            # bypass RandomSample.__getitem__ (which would draw a sample)
            frozen = dict.__getitem__(rs, target)
            lo = frozen.support()[0]
            assert lo >= 0, f"{anchor}-{target} distance support starts at {lo}"


# --- H3: a growth-sampled bond length must stay inside the pair's bonding cutoff ----
# Growth attaches a cation to an anion only (choose_atom_idx_to_attach_to). A placement is
# committed at the sampled bond length, but the coordination graph adds an edge only when
# distance < the pair cutoff. The Al-O sampler used to draw up to 2.2 A while the (O,Al)
# cutoff is 2.1, so ~25% of Al-O placements were committed yet counted as non-bonded,
# corrupting every CN-driven decision downstream (anchor weighting, saturation, charge).

def test_growth_sample_bond_lengths_within_bonding_cutoff():
    from default_constants import sample_dist, pair_cutoffs, OXIDATION_POS, OXIDATION_NEG

    cation, anion = set(OXIDATION_POS), set(OXIDATION_NEG)
    checked = 0
    for anchor, rs in sample_dist.items():
        for added in list(rs.keys()):
            # growth only ever samples a cation<->anion pair; skip same-class entries
            # (Si-Si, Al-Al, Si-Al, O-O), which growth never draws.
            if (anchor in cation) == (added in cation):
                continue
            frozen = dict.__getitem__(rs, added)        # bypass the sampling __getitem__
            upper = float(frozen.support()[1])
            cutoff = pair_cutoffs[(anchor, added)]
            # `<=` (not `<`): scipy uniform's support is half-open [loc, loc+scale), so a
            # nominal upper bound equal to the cutoff still yields draws strictly below it
            # (e.g. the Al-O window [1.7, 2.1) against the 2.1 cutoff).
            assert upper <= cutoff, (
                f"{anchor}-{added}: sample max {upper} exceeds bonding cutoff {cutoff}; "
                "a placed atom could be counted as non-bonded")
            checked += 1
    assert checked >= 4, "expected the Si-O, O-Si, Al-O, O-Al growth pairs to be checked"
