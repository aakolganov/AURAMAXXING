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
