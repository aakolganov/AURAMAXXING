"""Tests for the VdW surface area + cap-group areal-concentration metric (stats/surface.py).

The area is a periodic Shrake–Rupley (SASA) sampling; analytic cases have exact answers, so these
assert against them directly (with generous tolerances for the point-sampling noise).
"""
import numpy as np
import pytest

from stats.surface import (sasa_surface_area, surface_area, cap_areal_density,
                           vdw_radius, vdw_radii_array)

BIG = (40.0, 40.0, 40.0)


# --- radius model ------------------------------------------------------------------

def test_vdw_radius_sources():
    from ase.data import atomic_numbers, vdw_radii
    assert vdw_radius("O") == pytest.approx(1.52, abs=0.01)      # ASE table
    # Ti has NaN in ase.data.vdw_radii -> curated Alvarez-2013 supplement (finite, ~2.11)
    assert not np.isfinite(vdw_radii[atomic_numbers["Ti"]])
    assert vdw_radius("Ti") == pytest.approx(2.11, abs=1e-6)
    # probe adds linearly; explicit override wins over every table
    assert vdw_radius("O", probe=1.4) == pytest.approx(1.52 + 1.4, abs=0.01)
    assert vdw_radius("O", overrides={"O": 9.9}) == 9.9


def test_vdw_radius_covalent_fallback_warns():
    # an element with neither an ASE VdW radius nor a supplement entry -> covalent × 1.2, with a warning
    from ase.data import atomic_numbers, vdw_radii, covalent_radii
    from stats.surface import _COV_FALLBACK_FACTOR, _VDW_SUPPLEMENT, _warned_fallback
    sym = "Ru"
    assert not np.isfinite(vdw_radii[atomic_numbers[sym]]) and sym not in _VDW_SUPPLEMENT
    _warned_fallback.discard(sym)
    with pytest.warns(UserWarning, match="van-der-Waals"):
        r = vdw_radius(sym)
    assert r == pytest.approx(covalent_radii[atomic_numbers[sym]] * _COV_FALLBACK_FACTOR, abs=1e-9)


def test_vdw_radii_array_maps_per_atom():
    r = vdw_radii_array(["O", "H", "O"])
    assert r[0] == r[2] == pytest.approx(1.52, abs=0.01)
    assert r[1] == pytest.approx(1.20, abs=0.01)


# --- analytic surface-area cases ---------------------------------------------------

def test_single_sphere_area():
    r = vdw_radius("O")
    a = sasa_surface_area(np.array([[20.0, 20.0, 20.0]]), ["O"], BIG, n_points=500)
    assert a == pytest.approx(4 * np.pi * r**2, rel=0.02)


def test_two_overlapping_spheres_lens_area():
    # union of two equal spheres radius r, centres d apart (d < 2r): area = 2 pi r (2r + d)
    r, d = vdw_radius("O"), 1.6
    pos = np.array([[20.0, 20.0, 20.0], [20.0 + d, 20.0, 20.0]])
    a = sasa_surface_area(pos, ["O", "O"], BIG, n_points=800)
    assert a == pytest.approx(2 * np.pi * r * (2 * r + d), rel=0.03)


def test_flat_monolayer_is_roughly_two_faces():
    # dense flat sheet, PBC in x,y -> exposed area ~ top + bottom = 2 Lx Ly. A voxel/SASA PBC-seam
    # bug would inflate this with a spurious side wall, so this is the key periodicity guard.
    L, sp = 16.0, 1.2
    cell = (L, L, 30.0)
    g = np.arange(0, L, sp)
    X, Y = np.meshgrid(g, g, indexing="ij")
    mono = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 15.0)])
    a = sasa_surface_area(mono, ["O"] * len(mono), cell, n_points=200)
    assert 2 * L * L <= a <= 2 * L * L * 1.25     # >= 2 flat faces, with modest atomic corrugation


def test_roughness_increases_area_over_projection():
    # the whole point of the metric: a corrugated surface has more area than its flat projection.
    L, sp = 20.0, 1.3
    cell = (L, L, 40.0)
    g = np.arange(0, L, sp)
    X, Y = np.meshgrid(g, g, indexing="ij")
    flat = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 20.0)])
    z = 20.0 + 2.5 * np.sin(2 * np.pi * X.ravel() / L)     # corrugated top
    rough = np.column_stack([X.ravel(), Y.ravel(), z])
    a_flat = sasa_surface_area(flat, ["O"] * len(flat), cell, n_points=300)
    a_rough = sasa_surface_area(rough, ["O"] * len(rough), cell, n_points=300)
    assert a_rough > a_flat > L * L            # rough > flat, and both exceed the projected face


def test_area_converges_in_n_points():
    pos = np.array([[20.0, 20.0, 20.0], [21.6, 20.0, 20.0], [20.0, 21.6, 20.0]])
    areas = [sasa_surface_area(pos, ["O"] * 3, BIG, n_points=n) for n in (100, 300, 900)]
    assert abs(areas[2] - areas[1]) < abs(areas[1] - areas[0]) + 1e-6   # successive diffs shrink


def test_pbc_edge_atoms_are_neighbours():
    # two atoms just across the x-seam must bury each other's facing points (area < 2 isolated).
    L = 6.0
    cell = (L, L, 40.0)
    pos = np.array([[0.1, 3.0, 20.0], [L - 0.1, 3.0, 20.0]])
    a = sasa_surface_area(pos, ["O", "O"], cell, n_points=800)
    two_isolated = 2 * 4 * np.pi * vdw_radius("O") ** 2
    assert a < two_isolated * 0.98             # mutual burial across the periodic boundary


def test_sub_ulp_negative_coordinate_does_not_crash():
    # A boundary atom at a sub-ULP-negative x makes `x % L == L` exactly, which cKDTree(boxsize=...)
    # rejects; the clamp must keep it strictly < L so a finite area is returned rather than a crash.
    L = 14.237
    pos = np.array([[-1e-16, 5.0, 8.0], [7.0, 6.0, 8.0]])
    assert (-1e-16) % L == L                    # confirm the pathological wrap
    a = sasa_surface_area(pos, ["O", "Si"], (L, L, 16.0), n_points=100)
    assert np.isfinite(a) and a > 0


# --- cap_areal_density arithmetic + struct wrapper ---------------------------------

def test_cap_areal_density_arithmetic(make_struct):
    s = make_struct(["Si", "O", "O"], [[10, 10, 10], [11.6, 10, 10], [8.4, 10, 10]],
                    cell=(20.0, 20.0, 20.0))
    out = cap_areal_density(s, {"O-H": 3, "Si-F": 1}, 4, n_points=200)
    area_nm2 = surface_area(s, n_points=200) / 100.0
    assert out["area_nm2"] == pytest.approx(area_nm2, rel=1e-9)
    assert out["total"] == pytest.approx(4 / area_nm2, rel=1e-9)
    assert out["per_type"]["O-H"] == pytest.approx(3 / area_nm2, rel=1e-9)
    assert out["counts"] == {"O-H": 3, "Si-F": 1} and out["n_caps"] == 4


# --- metrics integration ----------------------------------------------------------

def test_analyze_structure_gates_cap_density(make_struct):
    from stats.metrics import analyze_structure
    s = make_struct(["Si"], [[10, 10, 10]], cell=(20.0, 20.0, 20.0))
    assert analyze_structure(s, is_saturation=False)["cap_areal_density"] is None
    # saturation on but the surface sub-metric disabled -> still None
    assert analyze_structure(s, is_saturation=True,
                             surface_opts={"enabled": False})["cap_areal_density"] is None


def test_merge_pools_density_as_ratio_of_sums():
    # pooled density must be Σcaps/Σarea, NOT the mean of per-structure densities.
    from stats.metrics import merge_metrics
    m1 = {"n_atoms": 1, "n_fixed": 0, "composition": {}, "coordination": {}, "homo_distance": {},
          "element_anion_distance": {}, "tau4": {}, "saturation": None,
          "cap_areal_density": {"per_type": {"O-H": 4 / 1.0}, "total": 4 / 1.0, "area_nm2": 1.0,
                                "counts": {"O-H": 4}, "n_caps": 4}}
    m2 = {"n_atoms": 1, "n_fixed": 0, "composition": {}, "coordination": {}, "homo_distance": {},
          "element_anion_distance": {}, "tau4": {}, "saturation": None,
          "cap_areal_density": {"per_type": {"O-H": 6 / 3.0}, "total": 6 / 3.0, "area_nm2": 3.0,
                                "counts": {"O-H": 6}, "n_caps": 6}}
    pooled = merge_metrics([m1, m2])["cap_areal_density"]
    assert pooled["total"] == pytest.approx((4 + 6) / (1.0 + 3.0))    # 2.5, not mean(4, 2)=3.0
    assert pooled["per_type"]["O-H"] == pytest.approx(10 / 4.0)
    assert pooled["n_structures"] == 2
