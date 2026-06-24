from functools import lru_cache

from base.amorphous_structure import AmorphousStruc, Limits
from default_constants import d_min_max
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

def within_z_limits(trial_coord: np.ndarray, limits: Limits) -> bool:
    ix = int((trial_coord[0]) / limits.dx)
    iy = int((trial_coord[1]) / limits.dy)
    if not (0 <= ix < limits.nx and 0 <= iy < limits.ny):
        return False
    lo = limits.lower_lim[ix, iy]
    hi = limits.upper_lim[ix, iy]
    return lo <= trial_coord[2] <= hi


@lru_cache(maxsize=None)
def fibonacci_sphere(samples: int = 100) -> np.ndarray:
    """
    Generates evenly distributed points on a unit sphere.
    Returns an array of shape (samples, 3).

    The result depends only on ``samples`` and is reused every placement attempt,
    so it is memoized. Callers must treat the returned array as read-only (they
    only ever scale a copy of it), so the shared cached array is never mutated.
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])

    return np.array(points)


def build_placement_cache(amorphous_struct: AmorphousStruc) -> tuple:
    """Precompute per-element periodic cKDTrees for collision checks.

    Returns ``(cell_dims, trees)`` where ``trees`` maps each element present to a
    ``(cKDTree, global_index_array)`` pair built over that element's wrapped
    positions. The cache is valid only while the structure is unchanged (no atom
    added, removed or moved); reuse it across repeated placement attempts on the
    same structure to avoid rebuilding the trees on every attempt -- the growth
    retry loop tries many anchors against an otherwise-fixed structure.
    Assumes an orthogonal cell.
    """
    cell_dims = amorphous_struct.atoms.cell.cellpar()[:3]
    symbols = np.array(amorphous_struct.atoms.get_chemical_symbols())
    positions = amorphous_struct.atoms.positions % cell_dims
    trees = {}
    for element in np.unique(symbols):
        idx_global = np.where(symbols == element)[0]
        trees[element] = (cKDTree(positions[idx_global], boxsize=cell_dims), idx_global)
    return cell_dims, trees


def place_atom_sphere(
        amorphous_struct: AmorphousStruc,
        atom_type: str,
        idx_anchor: int,
        bond_length: float,
        num_samples: int = 100,
        cache: tuple | None = None,
    ) -> bool:
    """
    Placement with simultaneous spherical
    sampling and KDTree collision detection.

    Pass ``cache`` (from ``build_placement_cache``) to reuse the per-element
    collision trees across attempts on an unchanged structure; otherwise one is
    built for this call.
    """

    # Per-element collision trees: reuse the caller's cache (structure is fixed
    # across repeated attempts on the same anchor) or build one for this call.
    if cache is None:
        cache = build_placement_cache(amorphous_struct)
    cell_dims, trees = cache

    anchor_pos = amorphous_struct.atoms.positions[idx_anchor]

    # 1. Generate all candidate points at once, wrapped into the periodic cell so the
    # Z-limit grid lookup and the periodic cKDTree query below both see in-box
    # coordinates (otherwise boundary-crossing candidates are wrongly z-rejected and
    # an out-of-box position could be committed). Assumes an orthogonal cell.
    unit_sphere = fibonacci_sphere(num_samples)
    candidates = (anchor_pos + (unit_sphere * bond_length)) % cell_dims

    # 2. Filter candidates by Z-limits first (fast pre-filter)
    if hasattr(amorphous_struct, 'limits') and amorphous_struct.limits is not None:
        valid_z_mask = np.array([within_z_limits(c, amorphous_struct.limits) for c in candidates])
        candidates = candidates[valid_z_mask]

    if len(candidates) == 0:
        return False # All points violated Z-limits

    # 3. Spatial collision checks against the cached per-element trees
    anchor_symbol = amorphous_struct.atoms[idx_anchor].symbol
    is_valid = np.ones(len(candidates), dtype=bool)

    for element, (tree, idx_global) in trees.items():

        # Exact translation of the original 'is_correlty_positions' logic:
        if element == atom_type:
            # If same element, anything under dmax is bad (covers both too_close and mid_bad)
            exclusion_radius = d_min_max[element][atom_type][1]
        else:
            # If different element, only too_close (< dmin) is bad
            exclusion_radius = d_min_max[element][atom_type][0]

        # Query all candidates simultaneously
        collisions = tree.query_ball_point(candidates, r=exclusion_radius)

        # CRITICAL: never count the anchor as a clash -- it is the atom we are
        # bonding to. It only appears in its own element's tree, so locate it
        # there and drop it from the hit lists for that element.
        anchor_local = None
        if element == anchor_symbol:
            anchor_local = int(np.searchsorted(idx_global, idx_anchor))

        # Invalidate candidates that hit an obstacle (other than the anchor)
        for i, cols in enumerate(collisions):
            if anchor_local is not None:
                if any(c != anchor_local for c in cols):
                    is_valid[i] = False
            elif len(cols) > 0:
                is_valid[i] = False

    # 4. Filter to only the candidates that passed all checks
    final_candidates = candidates[is_valid]

    if len(final_candidates) == 0:
        return False # Sterically blocked

    # 5. Pick a valid placement and commit
    chosen_idx = amorphous_struct.rng.choice(len(final_candidates))
    chosen_pos = final_candidates[chosen_idx]

    amorphous_struct.commit_atom(atom_type, position=chosen_pos)
    return True


def place_atom_most_z_space(
        amorphous_struct: AmorphousStruc,
        atom_type: str,
    ) -> None:
    """
    Place the initial atom somewhere within the limits that has a large 
    amount of local surrounding volume.
    """
    limits: Limits = amorphous_struct.limits
    rng = amorphous_struct.rng

    # 1. Calculate raw Z-volume for each grid cell
    dz = limits.upper_lim - limits.lower_lim

    # 2. Calculate "Local Volume" by smoothing the dz map
    # sigma=2.0 averages over a radius of roughly ~2 grid cells.
    local_volume = gaussian_filter(dz, sigma=2.0)

    # 3. Pick a placement probabilistically from the top candidates
    # We select from the top 5% of the most spacious spots to allow 
    # for slight starting variation between different simulation runs.
    flat_indices = np.argsort(local_volume.ravel())
    top_n = max(1, int(0.05 * len(flat_indices))) 
    
    chosen_flat_idx = rng.choice(flat_indices[-top_n:])
    ix, iy = np.unravel_index(chosen_flat_idx, local_volume.shape)

    # 4. Convert grid indices back to physical 3D coordinates
    # Adding 0.5 places the atom in the exact middle of the X/Y grid cell
    x = (ix + 0.5) * limits.dx
    y = (iy + 0.5) * limits.dy
    z = limits.lower_lim[ix, iy] + 0.5 * dz[ix, iy]

    amorphous_struct.commit_atom(atom_type, position=np.array([x, y, z]))


def place_atom_force(
        amorphous_struct: AmorphousStruc,
        atom_type: str,
        idx_anchor: int,
        bond_length: float,
        num_samples: int = 100
    ) -> bool:
    """
    Placement with simultaneous spherical sampling.
    If no valid placement is found (steric clashes), picks the position
    that minimizes the overlap (least bad).
    """
    
    anchor_pos = amorphous_struct.atoms.positions[idx_anchor]

    # 1. Generate all candidate points at once, wrapped into the periodic cell so the
    # periodic cKDTree query sees in-box coordinates and we never commit an
    # out-of-box position. Assumes an orthogonal cell.
    cell_dims = amorphous_struct.atoms.cell.cellpar()[:3]
    unit_sphere = fibonacci_sphere(num_samples)
    candidates = (anchor_pos + (unit_sphere * bond_length)) % cell_dims

    symbols = np.array(amorphous_struct.atoms.get_chemical_symbols())
    positions = amorphous_struct.atoms.positions

    # Track the minimum margin for each candidate.
    # Margin = distance_to_nearest_obstacle - exclusion_radius.
    # Positive margin = valid. Negative margin = overlap.
    # We want to maximize this value.
    min_margins = np.full(len(candidates), np.inf)

    # Group by unique elements to minimize tree builds
    for element in np.unique(symbols):
        
        if element == atom_type:
            exclusion_radius = d_min_max[element][atom_type][1] 
        else:
            exclusion_radius = d_min_max[element][atom_type][0] 

        # Mask and fetch positions for this element
        elem_mask = (symbols == element)
        elem_mask[idx_anchor] = False 
        obs_pos = positions[elem_mask]

        if len(obs_pos) == 0:
            continue

        # Ensure positions are wrapped within the box for cKDTree
        obs_pos = obs_pos % cell_dims

        # Build tree with PBCs applied
        tree = cKDTree(obs_pos, boxsize=cell_dims)
        
        # Query nearest neighbor distance for all candidates
        dists, _ = tree.query(candidates, k=1)
        
        # Update minimum margins
        margins = dists - exclusion_radius
        min_margins = np.minimum(min_margins, margins)

    # 4. Select candidate
    # First try to pick from valid ones (margin >= 0)
    valid_indices = np.where(min_margins >= 0)[0]
    
    if len(valid_indices) > 0:
        chosen_idx = amorphous_struct.rng.choice(valid_indices)
    else:
        # Fallback: pick the "least bad" one (max margin)
        chosen_idx = np.argmax(min_margins)

    chosen_pos = candidates[chosen_idx]
    
    amorphous_struct.commit_atom(atom_type, position=chosen_pos)
    return True


def slice_structure(amorphous_struct: AmorphousStruc) -> None:
    """
    Removes all atoms from the structure that fall outside of the defined Z-limits
    for their given X, Y position. If no limits are defined, this function does nothing.
    """
    if amorphous_struct.limits is None or len(amorphous_struct) == 0:
        return

    limits = amorphous_struct.limits

    # It's good practice to wrap atoms into the cell before checking limits,
    # especially with periodic boundary conditions.
    amorphous_struct.atoms.wrap()
    positions = amorphous_struct.atoms.get_positions()

    # Vectorized calculation of grid indices for all atoms.
    # This assumes the limit grid's origin is at (0, 0).
    ix = np.floor(positions[:, 0] / limits.dx).astype(int)
    iy = np.floor(positions[:, 1] / limits.dy).astype(int)

    # 1. Identify atoms that are within the XY bounds of the grid.
    #    Atoms outside this grid are automatically considered "out of bounds".
    in_xy_grid_mask = (ix >= 0) & (ix < limits.nx) & (iy >= 0) & (iy < limits.ny)

    # Initialize a mask to keep atoms. Start with all False.
    keep_mask = np.zeros(len(amorphous_struct), dtype=bool)

    # 2. For atoms inside the XY grid, check if their Z-coordinate is valid.
    atoms_in_grid_indices = np.where(in_xy_grid_mask)[0]
    if atoms_in_grid_indices.size > 0:
        z_valid = positions[atoms_in_grid_indices, 2]
        lower_bounds = limits.lower_lim[ix[atoms_in_grid_indices], iy[atoms_in_grid_indices]]
        upper_bounds = limits.upper_lim[ix[atoms_in_grid_indices], iy[atoms_in_grid_indices]]
        in_z_bounds = (z_valid >= lower_bounds) & (z_valid <= upper_bounds)
        keep_mask[atoms_in_grid_indices] = in_z_bounds

    # 3. Never remove fixed atoms (e.g. a frozen substrate). They can legitimately
    #    sit outside the growth volume, and dropping them would also collapse the
    #    FixAtoms constraint so the substrate would no longer be held during finalize.
    keep_mask[list(amorphous_struct.fixed_indices())] = True

    # 4. Remove atoms that are not in the keep mask
    remove_mask = ~keep_mask
    if np.any(remove_mask):
        amorphous_struct.remove_atom(remove_mask)