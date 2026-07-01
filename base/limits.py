from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Limits:
    nx: int
    ny: int
    dx: float
    dy: float
    upper_lim: Optional[np.ndarray] = field(default=None, repr=False)
    lower_lim: Optional[np.ndarray] = field(default=None, repr=False)


def move_limits(amorphous_struct, move_by: float = 2, move_limit: str = "top") -> None:
    limits = amorphous_struct.limits
    match move_limit:
        case "top":
            limits.upper_lim += move_by
        case "bottom":
            limits.lower_lim -= move_by
        case "both":
            limits.upper_lim += move_by/2
            limits.lower_lim -= move_by/2
        case _:
            print("Unknown side to move the limit, defaulting to top")
            limits.upper_lim += move_by


def _install_limit(amorphous_struc, init_limit: np.ndarray, is_for: str, steps: int) -> None:
    """Attach ``init_limit`` as the ``is_for`` (``top``/``bottom``) growth boundary.

    Creates the ``Limits`` grid on first use (dx/dy from the cell and ``steps``); on a second
    call it fills in the still-missing side, requiring the same grid shape as the existing one.
    Shared by ``make_limits_fourier`` and ``make_limit_flat`` (only the ``init_limit`` differs).
    """
    if amorphous_struc.limits is None:
        dx = amorphous_struc.atoms.cell.cellpar()[0] / steps
        dy = amorphous_struc.atoms.cell.cellpar()[1] / steps
        match is_for:
            case "top":
                amorphous_struc.limits = Limits(upper_lim=init_limit, nx=steps, ny=steps, dx=dx, dy=dy)
            case "bottom":
                amorphous_struc.limits = Limits(lower_lim=init_limit, nx=steps, ny=steps, dx=dx, dy=dy)
    else:
        if init_limit.shape != (amorphous_struc.limits.nx, amorphous_struc.limits.ny):
            raise ValueError(
                f"the new '{is_for}' limit has shape {init_limit.shape}, but the existing "
                f"limits grid is ({amorphous_struc.limits.nx}, {amorphous_struc.limits.ny}); "
                "both sides must be built with the same 'steps'")
        match is_for:
            case "top":
                amorphous_struc.limits.upper_lim = init_limit
            case "bottom":
                amorphous_struc.limits.lower_lim = init_limit


def make_limits_fourier(
        amorphous_struc,
        z_av: float,
        alpha: float,
        is_for: str,
        n_max: int = 6,
        m_max: int = 6,
        steps: int = 250,
        ) -> None:

    from helpers.fourier_functions import make_fourier_function

    init_limit = make_fourier_function(
        amorphous_struc.atoms.cell.cellpar()[0],
        amorphous_struc.atoms.cell.cellpar()[1],
        steps,
        alpha,
        n_max,
        m_max,
        amorphous_struc.rng,
    )
    curr_z_av = np.mean(init_limit)
    init_limit += (z_av - curr_z_av)

    _install_limit(amorphous_struc, init_limit, is_for, steps)


def make_limit_flat(
        amorphous_struc,
        z_val: float,
        is_for: str,
        steps: int = 250,
    ) -> None:
    init_limit = np.full((steps, steps), z_val)
    _install_limit(amorphous_struc, init_limit, is_for, steps)


def fix_limits(limits: Limits, hard_limit: Optional[str]=None) -> None:
    if limits.upper_lim is None or limits.lower_lim is None:
        raise ValueError("Both limits are expected to be defined")

    if hard_limit in ["top", "upper"]:
        # Hard limit is top, so we adjust bottom to not exceed top
        limits.lower_lim = np.minimum(limits.lower_lim, limits.upper_lim)
    elif hard_limit in ["bottom", "lower"]:
        # Hard limit is bottom, so we adjust top to not be below bottom
        limits.upper_lim = np.maximum(limits.upper_lim, limits.lower_lim)
    else:
        # Default: swap values where they are inverted to ensure lower <= upper
        lower = limits.lower_lim
        upper = limits.upper_lim
        limits.lower_lim = np.minimum(lower, upper)
        limits.upper_lim = np.maximum(lower, upper)