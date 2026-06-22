"""Tests for the growth-boundary (Limits) helpers."""
import numpy as np

from base.limits import make_limit_flat, make_limits_fourier, fix_limits, move_limits
from helpers.atom_placing import within_z_limits


def _struct(make_struct):
    return make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))


def test_make_limit_flat_shape_and_values(make_struct):
    s = _struct(make_struct)
    make_limit_flat(s, z_val=5.0, is_for="bottom", steps=64)
    make_limit_flat(s, z_val=15.0, is_for="top", steps=64)
    assert s.limits.lower_lim.shape == (64, 64)
    assert np.allclose(s.limits.lower_lim, 5.0)
    assert np.allclose(s.limits.upper_lim, 15.0)
    assert s.limits.dx == 20.0 / 64


def test_within_z_limits(make_struct):
    s = _struct(make_struct)
    make_limit_flat(s, z_val=5.0, is_for="bottom")
    make_limit_flat(s, z_val=15.0, is_for="top")
    lim = s.limits
    assert within_z_limits(np.array([10.0, 10.0, 10.0]), lim)      # inside
    assert not within_z_limits(np.array([10.0, 10.0, 2.0]), lim)   # below
    assert not within_z_limits(np.array([10.0, 10.0, 18.0]), lim)  # above
    assert not within_z_limits(np.array([25.0, 10.0, 10.0]), lim)  # outside xy grid


def test_make_limits_fourier_centered_at_z_av(make_struct):
    s = _struct(make_struct)
    make_limits_fourier(s, z_av=18.0, alpha=0.1, is_for="top", steps=64)
    assert s.limits.upper_lim.shape == (64, 64)
    assert np.isfinite(s.limits.upper_lim).all()
    assert abs(s.limits.upper_lim.mean() - 18.0) < 1e-6


def test_fix_limits_hard_bottom_and_swap(make_struct):
    # inverted limits: lower(15) > upper(5)
    s = _struct(make_struct)
    make_limit_flat(s, z_val=15.0, is_for="bottom")
    make_limit_flat(s, z_val=5.0, is_for="top")
    fix_limits(s.limits, hard_limit="bottom")           # bottom fixed -> raise upper
    assert np.all(s.limits.upper_lim >= s.limits.lower_lim)
    assert np.allclose(s.limits.lower_lim, 15.0)

    s2 = _struct(make_struct)
    make_limit_flat(s2, z_val=15.0, is_for="bottom")
    make_limit_flat(s2, z_val=5.0, is_for="top")
    fix_limits(s2.limits)                                # default: swap so lower<=upper
    assert np.allclose(s2.limits.lower_lim, 5.0)
    assert np.allclose(s2.limits.upper_lim, 15.0)


def test_move_limits(make_struct):
    s = _struct(make_struct)
    make_limit_flat(s, z_val=5.0, is_for="bottom")
    make_limit_flat(s, z_val=15.0, is_for="top")
    move_limits(s, move_by=2.0, move_limit="top")
    assert np.allclose(s.limits.upper_lim, 17.0)
    move_limits(s, move_by=2.0, move_limit="bottom")
    assert np.allclose(s.limits.lower_lim, 3.0)
