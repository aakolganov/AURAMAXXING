"""Tests for the synthetic Fourier surface generator."""
import numpy as np
import pytest

from helpers.fourier_functions import make_fourier_function


def test_shape_and_finite():
    surf = make_fourier_function(20.0, 20.0, 64, 0.1, 6, 6, seed=0)
    assert surf.shape == (64, 64)
    assert np.isfinite(surf).all()


def test_reproducible_with_seed():
    a = make_fourier_function(20.0, 20.0, 64, 0.1, 6, 6, seed=3)
    b = make_fourier_function(20.0, 20.0, 64, 0.1, 6, 6, seed=3)
    assert np.allclose(a, b)


def test_smaller_alpha_is_rougher():
    # coefficient std ~ 1/sqrt(alpha*...), so smaller alpha -> larger amplitude
    rough = make_fourier_function(20.0, 20.0, 128, 0.01, 6, 6, seed=1).std()
    smooth = make_fourier_function(20.0, 20.0, 128, 1.0, 6, 6, seed=1).std()
    assert rough > smooth


@pytest.mark.parametrize("kwargs", [
    dict(Lx=0.0, Ly=20.0, steps=64, alpha=0.1, n_max=6, m_max=6),
    dict(Lx=20.0, Ly=20.0, steps=0, alpha=0.1, n_max=6, m_max=6),
    dict(Lx=20.0, Ly=20.0, steps=64, alpha=0.0, n_max=6, m_max=6),
    dict(Lx=20.0, Ly=20.0, steps=64, alpha=0.1, n_max=0, m_max=6),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        make_fourier_function(seed=0, **kwargs)
