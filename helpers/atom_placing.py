from base.amorphous_structure import AmorphousStruc, Limits
from default_constants import d_min_max, sample_dist, default_max_cn, pair_cutoffs
import numpy as np
from ase import Atom
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


# def is_correlty_positions(amorphous_struct: AmorphousStruc,
#                        idx_chunk: List[int],
#                        new_atom_type: str,
#                        new_position: np.ndarray) -> bool:
#         """
#         Vectorized validity check to check distances if we can place atom based on the distances
#         """
#         # 0 ) Check if even we need something to check:
#         if len(idx_chunk) == 0:
#             return False

#         # 1) Cashing all the necessary atoms
#         symbols = amorphous_struct.atoms.get_chemical_symbols()
#         positions = amorphous_struct.atoms.get_positions()
#         cell = amorphous_struct.atoms.get_cell()
#         pbc_flags = amorphous_struct.atoms.get_pbc()

#         # 2) Calculating all MIC distances:

#         dists, dmat = get_distances(
#             p1 = new_position.reshape(1,3),
#             p2 = positions[idx_chunk],
#             cell = cell,
#             pbc = pbc_flags,
#         )

#         dist = dmat[0] # (M,)

#         # 3) Collecting all the thresholds
#         dmin = np.array([
#             d_min_max[symbols[k]][new_atom_type][0]
#             for k in idx_chunk
#         ])
#         dmax = np.array([
#             d_min_max[symbols[k]][new_atom_type][1]
#             for k in idx_chunk
#         ])
#         same = np.array([
#             symbols[k] == new_atom_type
#             for k in idx_chunk
#         ])

#         # 4) Assesing the distances and if we are not connecting the same atom kinds
#         too_close = dist < dmin
#         mid_bad = (dist < dmax ) & (dist > dmin) & same
#         # too_far = dist > dmax # ATM, we're considering that if the atoms are too far - this is fine

#         # 5) If one of these conditions does not work - reject the placement
#         if np.any(too_close | mid_bad):
#             return False

#         return True


# def place_atom_random(
#         amorphous_struct: AmorphousStruc,
#         atom_type: str,
#         idx_anchor: int,
#         max_iters: int = 25,
#     ) -> bool:
    
#     anchor_atom = amorphous_struct.atoms[idx_anchor].symbol
#     anchor_position = amorphous_struct.atoms.get_positions()[idx_anchor]

#     for _ in range(max_iters):
#         trial_vector = amorphous_struct.rng.standard_normal(3)
#         trial_vector /= np.linalg.norm(trial_vector)
#         trial_dist = sample_dist[anchor_atom][atom_type]
#         trial_position = anchor_position + trial_vector * trial_dist
#         if within_z_limits(trial_position, amorphous_struct.limits):
#             new_position = trial_position
#             break
#     else:
#         return False
    
#     positions = amorphous_struct.atoms.get_positions()
#     z_tol = 2.8 # we are checking atoms within 2.8 Angs, so 1-2 layers
#     zdiff = np.abs(new_position[2] - positions[:, 2])
#     close_idxs = np.where(zdiff <= z_tol)[0]
#     if not is_correlty_positions(amorphous_struct, close_idxs, new_atom_type=atom_type, new_coords=new_position):
#         return False
    
#     amorphous_struct.commit_atom(atom_type, coords=new_position)
#     return True


def fibonacci_sphere(samples: int = 100) -> np.ndarray:
    """
    Generates evenly distributed points on a unit sphere.
    Returns an array of shape (samples, 3).
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


def place_atom_sphere(
        amorphous_struct: AmorphousStruc,
        atom_type: str,
        idx_anchor: int,
        bond_length: float,
        num_samples: int = 100
    ) -> bool:
    """
    Placement with simultaneous spherical 
    sampling and KDTree collision detection.
    """
    
    anchor_pos = amorphous_struct.atoms.positions[idx_anchor]

    # 1. Generate all candidate points at once
    unit_sphere = fibonacci_sphere(num_samples)
    candidates = anchor_pos + (unit_sphere * bond_length)

    # 2. Filter candidates by Z-limits first (fast pre-filter)
    if hasattr(amorphous_struct, 'limits') and amorphous_struct.limits is not None:
        valid_z_mask = np.array([within_z_limits(c, amorphous_struct.limits) for c in candidates])
        candidates = candidates[valid_z_mask]

    if len(candidates) == 0:
        return False # All points violated Z-limits

    # 3. Setup Spatial KDTree checks
    # Assuming orthogonal cell for boxsize PBCs
    cell_dims = amorphous_struct.atoms.cell.cellpar()[:3] 
    
    symbols = np.array(amorphous_struct.atoms.get_chemical_symbols())
    positions = amorphous_struct.atoms.positions
    is_valid = np.ones(len(candidates), dtype=bool)

    # Group by unique elements to minimize tree builds
    for element in np.unique(symbols):
        
        # Exact translation of your 'is_correlty_positions' logic:
        if element == atom_type:
            # If same element, anything under dmax is bad (covers both too_close and mid_bad)
            exclusion_radius = d_min_max[element][atom_type][1] 
        else:
            # If different element, only too_close (< dmin) is bad
            exclusion_radius = d_min_max[element][atom_type][0] 

        # Mask and fetch positions for this element
        elem_mask = (symbols == element)
        elem_mask[idx_anchor] = False # CRITICAL: Don't check against the anchor!
        obs_pos = positions[elem_mask]

        if len(obs_pos) == 0:
            continue

        # Ensure positions are wrapped within the box for cKDTree
        obs_pos = obs_pos % cell_dims

        # Build tree with PBCs applied
        tree = cKDTree(obs_pos, boxsize=cell_dims)
        
        # Query all candidates simultaneously
        collisions = tree.query_ball_point(candidates, r=exclusion_radius)
        
        # Invalidate candidates that hit an obstacle
        for i, cols in enumerate(collisions):
            if len(cols) > 0:
                is_valid[i] = False

    # 4. Filter to only the candidates that passed all checks
    final_candidates = candidates[is_valid]

    if len(final_candidates) == 0:
        return False # Sterically blocked

    # 5. Pick a valid placement and commit
    chosen_idx = amorphous_struct.rng.choice(len(final_candidates))
    chosen_pos = final_candidates[chosen_idx]
    
    # Note: Use the argument name your commit_atom method expects (e.g., position=chosen_pos)
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

    # 1. Generate all candidate points at once
    unit_sphere = fibonacci_sphere(num_samples)
    candidates = anchor_pos + (unit_sphere * bond_length)

    cell_dims = amorphous_struct.atoms.cell.cellpar()[:3] 
    
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

    # 3. Remove atoms that are not in the keep mask
    remove_mask = ~keep_mask
    if np.any(remove_mask):
        amorphous_struct.remove_atom(remove_mask)