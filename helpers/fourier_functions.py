import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
             of shape `(steps, steps)`.
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
    x = np.linspace(0, Lx, steps)
    y = np.linspace(0, Ly, steps)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="xy")

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

def make_fourier_function_const_V(Lx: float,
                          Ly: float,
                          steps: int,
                          alpha: float,
                          n_max: int,
                          m_max: int,
                          roughness_rms: Optional[float] = None,
                          H: float = 1.5,
                          seed: Optional[Union[int, np.random.Generator]] = None
                          ):

    """
    Generates a synthetic 2D Fourier series-based surface that ensures the constant volume under it

    This function constructs a 2D surface by superimposing sinusoidal terms of varying frequencies
    and random coefficients. The Fourier series coefficients are determined based on a specified
    attenuation factor (alpha), ensuring that higher-frequency terms contribute progressively
    less to the overall surface. A random seed or generator can be used for reproducibility of the
    surface generation process. Parameter H controls the basic height.

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
    :param roughness_rms: Optional roughness rms for the surface generation process
    :type roughness_rms: float.
    :param H: Basic Height of the surface domain.
    :type H: float
    :param seed: Optional random seed or generator for reproducibility. It can be an integer seed
                 or a numpy random generator instance.
    :type seed: Optional[Union[int, np.random.Generator]]
    :return: A 2D Fourier series-based surface represented as a NumPy array
             of shape `(steps, steps)`.
    :rtype: numpy.ndarray

    :raises ValueError: If any of the following conditions are met:
        - Lx or Ly is not positive.
        - steps is not a positive integer.
        - alpha is not positive.
        - n_max or m_max is not a positive integer.
    """
    # generates the fourier function
    surface = make_fourier_function(Lx, Ly, steps, alpha, n_max, m_max, seed)
    # nullify the mean value
    surface -= surface.mean()

    # 2) (optionally) Scaling roughness for needed RMS value:
    if roughness_rms is not None:
        current_rms = surface.std()
        if current_rms > 0:
            surface *= (roughness_rms / current_rms)
    # 3) Adding basic height H:
    surface += H

    return surface

def compute_volume(surface: np.ndarray, Lx: float, Ly: float) -> float:
    """
    estimation of the volume under the surface
    """
    return np.mean(surface) * (Lx * Ly)


def plot_volume_vs_alpha(alphas, Lx=1.0, Ly=1.0, steps=200, n_max=5, m_max=5):

    """
    helper function to plot volume vs alpha
    :param alphas: range of alphas (attenuation factors) to plot
    :param Lx: Length of the surface domain along the x-axis.
    :param Ly: Length of the surface domain along the y-axis.
    :param steps: How many discrete points (grid resolution) along each axis.
    :param n_max: Maximum mode for Fourier series terms along the x-axis.
    :param m_max: Maximum mode for Fourier series terms along the y-axis.
    :return: plot of volume vs alpha
    """

    volumes = []
    for α in alphas:
        surf = make_fourier_function_const_V(Lx, Ly, steps, α, n_max, m_max, seed=42)
        volumes.append(compute_volume(surf, Lx, Ly))

    plt.figure()
    plt.plot(alphas, volumes, marker='o')
    plt.xlabel(r'$\alpha$')
    plt.ylabel("Volume under surface")
    plt.title('V vs α')
    plt.grid(True)
    plt.show()


def plot_sample_surface(alpha=1.0, n_max=5, m_max=5, Lx=1.0, Ly=1.0, steps=100):

    """
    helper function to plot sample Fourier series surface
    :param alpha: Attenuation factor controlling the decay of Fourier coefficients based on
    :param n_max: Maximum mode for Fourier series terms along the x-axis.
    :param m_max: Maximum mode for Fourier series terms along the y-axis.
    :param Lx: Length of the surface domain along the x-axis.
    :param Ly: Length of the surface domain along the y-axis.
    :param steps: Maximum number of discrete points (grid resolution) along each axis.
    :return: Plot of sample Fourier series surface
    """

    surf = make_fourier_function_const_V(Lx, Ly, steps, alpha, n_max, m_max, seed=123)
    x = np.linspace(0, Lx, steps)
    y = np.linspace(0, Ly, steps)
    X, Y = np.meshgrid(x, y, indexing='xy')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, surf, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Sample surface (α={alpha}, n_max={n_max})')
    plt.show()

if __name__ == '__main__':
    # plot alpha vs volume
    alphas = np.linspace(0.0001, 2, 20)
    plot_volume_vs_alpha(alphas, Lx=21.5, Ly=21.5, steps=300, n_max=5, m_max=5)

    for a in alphas:
        plot_sample_surface(alpha=a, n_max=5, m_max=5, Lx=21.5, Ly=21.5, steps=300)




