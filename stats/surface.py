"""Van-der-Waals surface area + cap-group areal concentration for rough slabs.

Capping-group concentration is reported as **groups per nm²**, normalised by the *true*
van-der-Waals surface area of the slab -- not the projected ``Lx*Ly`` box face, which badly
under-counts a Fourier-roughened top (the real area is ``∬√(1+|∇z|²) dx dy >> Lx*Ly``).

The area is computed by a periodic-aware **Shrake–Rupley** surface sampling (SASA): each atom's
van-der-Waals sphere is sampled with evenly spread points, and a point that lies inside no other
atom's sphere is "exposed"; the exposed fraction times ``4π r²`` is that atom's contribution.
At ``probe = 0`` this is the bare VdW surface; a positive ``probe`` inflates every radius by the
probe radius, giving the solvent-accessible surface. Periodicity is x,y (vacuum in z), matching the
slab convention, via a minimum-image test. This was chosen over a voxel/marching-cubes isosurface in
a benchmark: SASA reproduced the analytic sphere/lens areas exactly (marching cubes carried a ~5%
curvature bias and needed a heavier ``scikit-image`` dependency), converged by ~200 sample points,
and is dependency-free (reuses the growth engine's ``fibonacci_sphere``).
"""
from __future__ import annotations

import warnings

import numpy as np
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from scipy.spatial import cKDTree

from helpers.atom_placing import fibonacci_sphere

# Van-der-Waals radii (Å) for the common oxide-former metals that ASE's ``vdw_radii`` table leaves
# undefined (NaN). Source: Alvarez, "A cartography of the van der Waals territories", Dalton Trans.
# 2013, 42, 8617 -- a whole-periodic-table tabulation.
_VDW_SUPPLEMENT: dict[str, float] = {
    "Sc": 2.15, "Ti": 2.11, "V": 2.07, "Cr": 2.06, "Fe": 2.04, "Y": 2.32, "Zr": 2.23,
    "Nb": 2.18, "Mo": 2.17, "La": 2.43, "Ce": 2.42, "Hf": 2.23, "Ta": 2.22, "W": 2.18,
}
# Last-resort fallback for an element with neither an ASE VdW radius nor a supplement entry:
# covalent radii run ~15-20% below VdW, so inflate by this factor. Emits a one-time warning.
_COV_FALLBACK_FACTOR = 1.2

_warned_fallback: set = set()


def vdw_radius(symbol: str, probe: float = 0.0, overrides: dict | None = None) -> float:
    """Van-der-Waals radius of ``symbol`` (Å), plus ``probe``.

    Resolution order: an explicit ``overrides[symbol]``; else ASE ``vdw_radii`` where defined; else
    the curated Alvarez-2013 supplement for the undefined oxide metals; else ``covalent_radii`` ×
    ``_COV_FALLBACK_FACTOR`` (with a one-time warning, since that is only an approximation).
    """
    if overrides is not None and symbol in overrides:
        return float(overrides[symbol]) + probe
    z = atomic_numbers[symbol]
    r = vdw_radii[z]
    if np.isfinite(r) and r > 0:
        return float(r) + probe
    if symbol in _VDW_SUPPLEMENT:
        return _VDW_SUPPLEMENT[symbol] + probe
    if symbol not in _warned_fallback:
        _warned_fallback.add(symbol)
        warnings.warn(
            f"no tabulated van-der-Waals radius for {symbol!r}; using covalent radius × "
            f"{_COV_FALLBACK_FACTOR} as a fallback (override via surface vdw_overrides).",
            stacklevel=2)
    return float(covalent_radii[z]) * _COV_FALLBACK_FACTOR + probe


def vdw_radii_array(symbols, probe: float = 0.0, overrides: dict | None = None) -> np.ndarray:
    """Per-atom VdW radii (+probe) for a symbol sequence; resolves each element once."""
    per_element = {s: vdw_radius(s, probe, overrides) for s in set(symbols)}
    return np.array([per_element[s] for s in symbols], dtype=float)


def sasa_surface_area(positions: np.ndarray, symbols, cell_dims, *, probe: float = 0.0,
                      n_points: int = 200, overrides: dict | None = None) -> float:
    """Total exposed VdW surface area (Å²) via periodic Shrake–Rupley sampling.

    ``positions`` (N×3, Å), ``symbols`` (length N), ``cell_dims`` = ``(Lx, Ly, Lz)``. The surface is
    periodic in x,y and open (vacuum) in z, so only the top and bottom faces are exposed -- there are
    no side faces. ``n_points`` sphere-sample points per atom trade accuracy for speed (~200 is
    converged for these slabs). ``probe`` defaults to 1.84 Å (BET-N₂ gauge); ~1.4 Å is water-accessible; 0 is the
    bare VdW surface (counts every crevice and internal void, comparable to no experiment).
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n == 0:
        return 0.0
    Lx, Ly, _ = (float(c) for c in cell_dims)
    r = vdw_radii_array(symbols, probe, overrides)
    r_max = float(r.max())

    # cKDTree(boxsize=[Lx, Ly, 0]) is x,y-periodic and z-free (0 = non-periodic axis). Periodic-axis
    # coordinates must lie in [0, L); z is left untouched.
    wpos = positions.copy()
    wpos[:, 0] %= Lx
    wpos[:, 1] %= Ly
    # A sub-ULP-negative coordinate makes ``x % L`` round up to exactly L, which cKDTree(boxsize=...)
    # rejects ("greater than the size of the periodic box"). Clamp strictly below L: nextafter(L, 0)
    # is the largest float < L, so this nudges only that pathological ==L case and is a no-op on every
    # in-box coordinate.
    wpos[:, 0] = np.minimum(wpos[:, 0], np.nextafter(Lx, 0.0))
    wpos[:, 1] = np.minimum(wpos[:, 1], np.nextafter(Ly, 0.0))
    tree = cKDTree(wpos, boxsize=[Lx, Ly, 0.0])
    unit = fibonacci_sphere(n_points)   # (n_points, 3), shared read-only cache

    total = 0.0
    for i in range(n):
        # Any atom that can bury a point on atom i's sphere has its centre within r_i + r_max of
        # centre i (a point is r_i from centre i, and buried needs centre_j within r_j <= r_max).
        cand = [j for j in tree.query_ball_point(wpos[i], r=r[i] + r_max) if j != i]
        full = 4.0 * np.pi * r[i] ** 2
        if not cand:
            total += full
            continue
        pts = wpos[i] + r[i] * unit                      # (n_points, 3)
        delta = pts[:, None, :] - wpos[cand][None, :, :]  # (n_points, m, 3)
        delta[..., 0] -= Lx * np.round(delta[..., 0] / Lx)   # x,y minimum image
        delta[..., 1] -= Ly * np.round(delta[..., 1] / Ly)
        buried = ((delta * delta).sum(-1) < (r[cand][None, :] ** 2)).any(axis=1)
        total += (np.count_nonzero(~buried) / n_points) * full
    return float(total)


def surface_area(struct, *, probe: float = 0.0, n_points: int = 200,
                 overrides: dict | None = None) -> float:
    """Total exposed VdW surface area (Å²) of an ``AmorphousStruc``'s atoms (all atoms contribute,
    including any fixed substrate). Thin wrapper over :func:`sasa_surface_area`."""
    atoms = struct.atoms
    return sasa_surface_area(atoms.get_positions(), atoms.get_chemical_symbols(),
                             atoms.cell.cellpar()[:3], probe=probe, n_points=n_points,
                             overrides=overrides)


def cap_areal_density(struct, counts: dict, n_caps: int, *, probe: float = 0.0,
                      n_points: int = 200, overrides: dict | None = None) -> dict:
    """Areal concentration of surface caps (groups per nm²), normalised by the total exposed VdW
    surface area. ``counts`` maps a cap-bond label (``"{anchor}-{cap}"``, e.g. ``"O-H"``) to its
    count; ``n_caps`` is the total. Cap detection lives in ``stats.metrics`` (single source of truth);
    this layer only computes the area and the density. Areas and counts are both stored so a pooled
    density can be formed as Σcounts / Σarea (see ``merge_metrics``)."""
    area_A2 = surface_area(struct, probe=probe, n_points=n_points, overrides=overrides)
    area_nm2 = area_A2 / 100.0   # 1 nm² = 100 Å²
    out = {"per_type": {}, "total": 0.0, "area_nm2": float(area_nm2),
           "counts": dict(counts), "n_caps": int(n_caps), "probe": float(probe)}
    if area_nm2 > 0:
        out["per_type"] = {label: c / area_nm2 for label, c in counts.items()}
        out["total"] = n_caps / area_nm2
    return out
