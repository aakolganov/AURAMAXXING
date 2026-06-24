import numpy as np
from typing import Optional, Union


def make_fourier_function(Lx: float,
                          Ly: float,
                          steps: int,
                          alpha: float,
                          n_max: int,
                          m_max: int,
                          seed: Optional[Union[int, np.random.Generator]] = None
                          ):
    """
    Generates a synthetic 2D Fourier series-based surface.

    This function constructs a 2D surface by superimposing sinusoidal terms of varying frequencies
    and random coefficients. The Fourier series coefficients are determined based on a specified
    attenuation factor (alpha), ensuring that higher-frequency terms contribute progressively
    less to the overall surface. A random seed or generator can be used for reproducibility of the
    surface generation process.

    :param Lx: Length of the surface domain along the x-axis.
    :type Lx: float
    :param Ly: Length of the surface domain along the y-axis.
    :type Ly: float
    :param steps: Number of discrete points (grid resolution) along each axis.
    :type steps: int
    :param alpha: Attenuation factor controlling the decay of Fourier coefficients based on
                  frequency magnitude.
    :type alpha: float
    :param n_max: Maximum mode for Fourier series terms along the x-axis.
    :type n_max: int
    :param m_max: Maximum mode for Fourier series terms along the y-axis.
    :type m_max: int
    :param seed: Optional random seed or generator for reproducibility. It can be an integer seed
                 or a numpy random generator instance.
    :type seed: Optional[Union[int, np.random.Generator]]
    :return: A 2D Fourier series-based surface represented as a NumPy array
             of shape `(steps, steps)`, indexed ``surface[ix, iy]`` (axis 0 = x, axis 1 = y).
    :rtype: numpy.ndarray

    :raises ValueError: If any of the following conditions are met:
        - Lx or Ly is not positive.
        - steps is not a positive integer.
        - alpha is not positive.
        - n_max or m_max is not a positive integer.
    """

    # ————— Input validation —————
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must be positive.")
    if steps <= 0 or not isinstance(steps, int):
        raise ValueError("steps must be a positive integer.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if n_max <= 0 or not isinstance(n_max, int):
        raise ValueError("n_max must be a positive integer.")
    if m_max <= 0 or not isinstance(m_max, int):
        raise ValueError("m_max must be a positive integer.")

    # ————— Random generator —————
    rng = np.random.default_rng(seed)

    # ————— 1D coordinate grids —————
    # endpoint=False so the node spacing is exactly Lx/steps (== Limits.dx), matching how the
    # z-limit lookups map a coordinate to a grid cell via int(x / dx).
    x = np.linspace(0, Lx, steps, endpoint=False)
    y = np.linspace(0, Ly, steps, endpoint=False)
    # indexing="ij" so surface[i, j] is the height at physical (x[i], y[j]); the limit lookups
    # index it as lim[ix, iy] with ix from x and iy from y, so axis 0 must be x, axis 1 must be y.
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")

    # ————— Precomputed angular grids —————
    X = (np.pi / Lx) * mesh_x  # shape (steps, steps)
    Y = (np.pi / Ly) * mesh_y  # shape (steps, steps)

    ### ————— Precompute sine terms for each modes —————
    # Sx[m] = sin((m+1) * X), Sy[n] = sin((n+1) * Y)
    m_indices = np.arange(1, m_max + 1) #mode indices
    n_indices = np.arange(1, n_max + 1)
    Sx = np.sin(m_indices[:, None, None] * X[None, :, :])  # shape (m_max, steps, steps)
    Sy = np.sin(n_indices[:, None, None] * Y[None, :, :])  # shape (n_max, steps, steps)

    # Generate random coefficients with appropriate std dev
    # std[n,m] = 1/sqrt(alpha * (n^2 + m^2))
    nn, mm = np.meshgrid(n_indices, m_indices, indexing='ij')  # nn, mm shape (n_max, m_max)
    std = 1.0 / np.sqrt(alpha * (nn**2 + mm**2)) #calculated standard deviation
    coeffs = rng.normal(loc=0.0, scale=std)  # shape (n_max, m_max)

    # Compute Fourier series via Einstein summation
    # F[i,j] = sum_n sum_m coeffs[n,m] * Sx[m,i,j] * Sy[n,i,j]
    surface = np.einsum('nm,mij,nij->ij', coeffs, Sx, Sy)
    return surface
