from typing import List, Tuple, Sequence
from numpy.typing import ArrayLike
import copy
from ase import Atoms, Atom
from ase.neighborlist import NeighborList, natural_cutoffs
from helpers.fourier_functions import *
from ase.geometry import get_distances
from default_constants import d_min_max, sample_dist, default_max_cn, pair_cutoffs
from ase.constraints import FixAtoms


class AmorphousStrucASE:
    """
    Represents an amorphous structure using the ASE (Atomic Simulation Environment) Atoms object.

    This class assists in creating, configuring, and analyzing structures with advanced functionalities
    such as periodic boundary conditions, neighbor lists, coordination number calculations, and more. Users can
    initialize the class with an existing ASE Atoms object, or define atomic symbols and positions for a custom build.

    Various customization options such as unit cell, periodic boundary conditions, and random seed generator
    are available. The class extends ASE capabilities with additional methods for exploring the structural
    properties of atoms, coordination environments, generation of structural constraints, and atom selections
    based on various criteria.

    :ivar atoms: ASE Atoms object representing the atomic structure.
    :type atoms: Atoms
    :ivar max_cn: Default settings for maximum coordination numbers.
    :type max_cn: dict
    :ivar cut_offs: Default cutoff distances for neighbors based on atom types.
    :type cut_offs: dict
    :ivar rng: Random number generator for stochastic operations.
    :type rng: numpy.random.Generator
    """
    def __init__(self,
                 atoms: Optional[Atoms] = None,
                 symbols: Optional[Sequence[str]] = None,
                 positions: Optional[ArrayLike] = None,
                 cell: Optional[ArrayLike] = None,
                 pbc: Union[bool, Sequence[bool]] = False,
                 seed: Optional[Union[int, np.random.Generator]] = None):
        """
        Initialize the ASE Atoms container.

        You can provide either:
          • atoms: an existing ASE Atoms object, or
          • symbols & positions to build one.
        If neither is provided, an empty Atoms object is created.

        Optional:
          – cell: 3×3 array or list for the unit cell
          – pbc: boolean or tuple of 3 booleans for periodic boundaries
        """
        # 1) If user passed an Atoms, copy it
        if atoms is not None:
            self.atoms = atoms.copy()

        # 2) If they passed symbols+positions, build from those
        elif (symbols is not None) and (positions is not None):
            self.atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=cell if cell is not None else None,
                pbc=pbc if pbc is not None else False
            )

        # 3) Otherwise, create a brand‐new empty Atoms
        else:
            self.atoms = Atoms()
            # apply cell if specified
            if cell is not None:
                self.atoms.set_cell(cell)
            # apply pbc if specified
            if pbc is not None:
                self.atoms.set_pbc(pbc)


        self.max_cn = default_max_cn.copy()
        self.cut_offs=pair_cutoffs  .copy()
        # If the user passes in a Generator, use it; otherwise build one from the seed.
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

        self.frozen_indices = []
        self._has_frozen = False

    def set_limits(self, alpha: float, n_m: int, const_V: Optional=False, H: Optional=0) -> None:
        """
        Sets the upper and lower limits of between the structure is allowed to grow.
        Current behavious is to make a stochastic fourier series specified through alpha and n_m for upper limit
        and setting the lower limit to 0, so the bottom surface is always more and less planar

        --- input ---
          alpha - empirical roughenss parameter of the Fourier function
          n_m - the number of Fouier made to use when making the function
          const_V - whether to use constant volume functionality or not.
          H - base height for const_V functionality.

        --- effect ---
          sets the class variable "limits" with a list containing, in this order:
            1D array of x-values used to made the function
            1D array of y-values used to made the function
            2D array of lower limit
            2D array of the upper limit
        """

        n_points = 500 # number of grid points
        x_vals = np.linspace(0, self.atoms.cell.cellpar()[0], num=n_points)
        y_vals = np.linspace(0, self.atoms.cell.cellpar()[1], num=n_points)
        self._x0, self._dx = x_vals[0], x_vals[1] - x_vals[0]
        self._y0, self._dy = y_vals[0], y_vals[1] - y_vals[0]
        self._nx, self._ny = len(x_vals), len(y_vals)

        init_upper_limit = make_fourier_function(self.atoms.cell.cellpar()[0], self.atoms.cell.cellpar()[1], 500, alpha, n_m, n_m, self.rng)
        if len(self.atoms) == 0: #generating low and high limit for empty structures
            if const_V: # if we are using constant volume functionality
                upper_limit = make_fourier_function_const_V(
                    Lx=self.atoms.cell.cellpar()[0],
                    Ly=self.atoms.cell.cellpar()[1],
                    steps=500,
                    alpha=alpha,
                    n_max=n_m,
                    m_max=n_m,
                    seed = self.rng,
                    H=H
                )

            shift = 1.5 # helper value to divide the Fourier function by positive and negative part
            positive_portion = init_upper_limit.copy() + shift
            positive_portion[positive_portion < 0] = 0
            negative_portion = init_upper_limit.copy() - shift
            negative_portion[negative_portion > 0] = 0
            negative_portion = np.abs(negative_portion)

            roughness_positive = np.mean((positive_portion - np.mean(positive_portion)) ** 2)
            roughness_negative = np.mean((negative_portion - np.mean(negative_portion)) ** 2)
            if roughness_positive > roughness_negative:

                upper_limit = positive_portion
            else:
                upper_limit = negative_portion

            lower_limit = np.full((500, 500), 0.0)
            self.limits = [x_vals, y_vals, lower_limit, upper_limit]

        else: # if initial structure is provided and we to grow on top of it
            #1) calculate the limits for the base structure
            z_max = float(self.atoms.positions[:, 2].max())

            #2) Bottot XY plane for the new structure = XY = z_max + 1.5 A
            extra_shift = 1.5
            baseline = z_max + extra_shift
            lower_limit = np.full((n_points, n_points), baseline)

            #3) Top limit = baseline + Fourier series
            shift = 0.0
            pos = init_upper_limit.copy()
            pos[pos < 0] = 0.0
            neg = init_upper_limit.copy()
            neg[neg > 0] = 0.0
            neg = np.abs(neg)
            # Again, selecting which relative value is higher
            if np.mean((pos - pos.mean()) ** 2) > np.mean((neg - neg.mean()) ** 2):
                upper_relative = pos
            else:
                upper_relative = neg
            upper_limit_on_base = baseline + upper_relative
            self.limits = [x_vals, y_vals, lower_limit, upper_limit_on_base]





    def copy(self) -> 'AmorphousStrucASE':
        """Deep-copy the structure."""
        return copy.deepcopy(self)

    def count_atoms(self, symbol: str) -> int:
        """counts atoms of certain type"""
        return self.atoms.get_chemical_symbols().count(symbol)

    def _pbc_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Helper function to return the minimum‐image distance between Cartesian coords a and b,
        given this Atoms object's cell & pbc flags.
        """
        dr = b - a
        if any(self.atoms.get_pbc()):
            cell = self.atoms.get_cell()
            inv = np.linalg.inv(cell)
            frac = inv.dot(dr)
            frac = frac - np.rint(frac)
            dr = cell.dot(frac)
        return np.linalg.norm(dr)


    def _build_neighbour_list(self) -> None:
        """
        Build ASE NeighborList from self.atoms and self.cut_offs.
        Expects self.cut_offs to be a dict mapping each symbol to a dict
        of cutoff distances to other symbols, e.g.:
        """
        if len(self.atoms) == 0:
            self._nl = None
            return

        # Get the list of atom symbols in order
        symbols = self.atoms.get_chemical_symbols()
        radii = natural_cutoffs(self.atoms)
        # Build the ASE NeighborList
        self._nl = NeighborList(radii, self_interaction=False, bothways=True)
        # Update with the current Atoms object
        self._nl.update(self.atoms)

    def update_atoms(self, atoms: Atoms):
        """If  self.atoms wholesale are changed, call this."""
        self.atoms = atoms

        if self._has_frozen and self.frozen_indices:
            #re-invoke frozen indices
            new_valid = []
            N = len(self.atoms)
            for i in self.frozen_indices:
                if 0 <= i < N:
                    new_valid.append(i)
            self.frozen_indices = new_valid

            # then add FixAtoms on these indices
            if self.frozen_indices:
                constraint = FixAtoms(indices=self.frozen_indices)
                self.atoms.set_constraint(constraint)

        self._build_neighbour_list()

    def get_cn(self, idx: int) -> int:
        """Coordination number = number of neighbors of atom `idx`."""
        if self._nl is None:
            return 0

        assert 0 <= idx < len(self.atoms), f"Trying to access an atom which does not exist, {idx=}"
        nbrs, _ = self._nl.get_neighbors(idx)
        if len (nbrs) ==0:
            return 0
        return len(nbrs)


    def set_i(self, atom_symbol: str, weight_z: bool = False, al_penalty: float=10E-8) -> int:

        """
        Pick an existing atom *of type* atom_symbol,
        whose coordination < max_cn, weighted by CN (and optionally by z).
        If atom_symbol == "Al", then candidates that already have an Al neighbor
        get their weight multiplied by `al_penalty` (default 10E-8).

        Parameters
        ----------
        atom_symbol : str
            Symbol of atom to bond to next.
        weight_z : bool, optional
            If True, further weight candidates by how central they are in Z.
        al_penalty : float, optional
            Multiplier for candidates that already have an Al neighbor (only when atom_symbol == "Al").

        Returns
        -------
        idx : int
            Index of chosen atom. g
        """

        # 0) ensure neighbor list up-to-date
        self._build_neighbour_list()

        # 1) get the vectors of all symbols and coordinates
        symbols = np.array(self.atoms.get_chemical_symbols())
        all_cn = np.array([self.get_cn(i) for i in range(len(symbols))])

        # 2) mask: right symbol and not saturated
        mask = (symbols == atom_symbol) & (all_cn < self.max_cn[atom_symbol])
        cand = np.nonzero(mask)[0]

        # 3) if none left, pick uniformly at random
        if cand.size == 0:
            return int(self.rng.integers(len(self.atoms)))

        # 4) group by coordination number
        cand_cns = all_cn[cand]
        unique_cns = np.unique(cand_cns)
        # prefer highest CN
        weights = 2 ** (unique_cns + 1)
        probs = weights/weights.sum()
        pick_cn = self.rng.choice(unique_cns, p=probs)

        #subset of indexes to attach
        sub = cand[cand_cns == pick_cn]

        #base weights = 1
        w = np.ones(len(sub), dtype=float)

        # 5)  if we want to weight also by z-coordinate:
        if weight_z:
            zpos = self.atoms.get_positions()[sub, 2]
            w = np.exp(-((zpos - zpos.mean()) ** 2))
            w = np.ones_like(zpos) if np.allclose(w, 0) else w / w.sum()

        # 6) If Al — penalty for already existing Al neighbors:
        if atom_symbol == "Al":
            al_indices = [i for i, s in enumerate(symbols) if s == "Al"]
            for i_sub, idx in enumerate (sub):
                #checking if there is Al within 3 angs distance
                pos_idx = self.atoms.get_positions()[idx]
                found_close_al = False
                for j in al_indices:
                    if j == idx:
                        continue # do not count self
                    d = self._pbc_dist(pos_idx, self.atoms.get_positions()[j])
                    d_cutoff = 3.5 # sphere cutoff distance
                    if d < d_cutoff:
                        found_close_al = True
                        break
                if found_close_al: #if we found Al close - multiplying the penalty by the weight of the Al neighbors
                    w[i_sub] *= al_penalty

        # 7) renormalize & choose
        total = w.sum()
        if total <= 0:
            # fallback to uniform
            w = np.ones_like(w)
            total = w.sum()
        w /= total

        return int(self.rng.choice(sub, p=w))

    def choose_vector(self,
                      atom_type: str,
                      idx_anchor: int,
                      limits: Optional = None,
                      max_iter: int = 250
                      ) -> Optional[np.ndarray]:
        """
        Propose a 3D position for an atom of type `atom_type` bonded to
        the existing atom `idx_anchor`.

        If `limits` is None, returns a single random placement at the ideal
        bond length. Otherwise retries up to `max_iter` times until the
        new z‐coordinate lies within the slab defined by `limits`, then
        applies PBC wrapping if needed.

        Parameters
        ----------
        atom_type : str
            Symbol of the new atom.
        idx_anchor : int
            Index of the atom to bond to.
        limits : tuple of four ndarrays, optional
            (x_bins, y_bins, down_limits, up_limits), defining at each grid
            cell (ix,iy) the allowed z‐range:
                z_min - down_limits[ix,iy] <= z <= z_min + up_limits[ix,iy].
        max_iter : int
            Number of sampling attempts before giving up.

        Returns
        -------
        coords : ndarray (3,) or None
            New Cartesian coordinates, or None if no valid placement found.
        """

        #grid parameters
        pos = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()
        anchor = pos[idx_anchor]
        dist = sample_dist[symbols[idx_anchor]][atom_type]

        #if no vertical limits, one shot random placement
        if limits is None:
            v = self.rng.standard_normal(3)
            v /= np.linalg.norm(v)
            return anchor + v * dist

        # if there are limits

        # 0) Check if there are PBC and adjust is needed
        has_pbc = any(self.atoms.get_pbc())
        cell = self.atoms.get_cell()
        inv_cellT = np.linalg.inv(cell.T)

        # 1) precompute the reference plane
        z_min = pos[:, 2].min()
        x_bins, y_bins, down, up = limits

        # 2) main loop
        for _ in range(max_iter):
            v = self.rng.standard_normal(3)
            v /= np.linalg.norm(v)
            new = anchor + v * dist

            # Check in we are in the xy limits

            ix = int((new[0] - self._x0) / self._dx)
            iy = int((new[1] - self._y0) / self._dy)

            if not (0 <= ix < self._nx and 0 <= iy < self._ny):
                continue #if we didn't hit the grid - try again

            lo = z_min - down[ix, iy]
            hi = z_min + up[ix, iy]

            if lo <= new[2] <= hi:
                # if PBC is active - then wrap it into the cell
                if has_pbc:
                    frac = inv_cellT.dot(new)
                    new = (frac - np.floor(frac)) @ cell
                return new
            # if we didn't succeed
        return None

    def check_placement(self,
                        idx_chunk: List[int],
                        new_atom_type: str,
                        new_coords: np.ndarray) -> bool:
        """
        Vectorized validity check to check distances if we can place atom based on the distances
        """
        #0 ) Check if even we need something to check:
        if len(idx_chunk) == 0:
            return False

        # 1) Cashing all the necessary atoms
        symbols = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        pbc_flags = self.atoms.get_pbc()

        # 2) Calculating all MIC distances:

        dists, dmat = get_distances(
            p1 = new_coords.reshape(1,3),
            p2 = positions[idx_chunk],
            cell = cell,
            pbc = pbc_flags,
        )

        dist = dmat[0] # (M,)

        # 3) Collecting all the thresholds
        dmin = np.array([
            d_min_max[symbols[k]][new_atom_type][0]
            for k in idx_chunk
        ])
        dmax = np.array([
            d_min_max[symbols[k]][new_atom_type][1]
            for k in idx_chunk
        ])
        same = np.array([
            symbols[k] == new_atom_type
            for k in idx_chunk
        ])

        # 4) Assesing the distances and if we are not connecting the same atom kinds
        too_close = dist < dmin
        mid_bad = (dist < dmax ) & (dist > dmin) & same
        #too_far = dist > dmax # ATM, we're considering that if the atoms are too far - this is fine

        # 5) If one of these conditions does not work - reject the placement
        if np.any(too_close | mid_bad):
            return False

        return True

    def _commit_atom(self, atom_type: str, coords: np.ndarray) -> None:
        """
        Simple wrap to add an atom and update the structure
        """
        self.atoms.append(Atom(atom_type, position=coords))
        self.update_atoms(self.atoms)

    def place_atom_random(
            self,
            atom_type: str,
            idx_anchor: Optional[int] = None,
            limits: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
            max_iter: int = 25
        ) -> bool:
        """
        Place a new atom of `atom_type` either at the center of the box
        (if idx_connect_to is None) or bonded to atom `idx_connect_to`,
        respecting optional vertical `limits`.

        Returns True if placement succeeded, False otherwise.
        """
        # 1) center‐of‐cell placement when no anchor
        if idx_anchor is None:
            # fractional [0.5,0.5,0.5] → Cartesian
            new_coords =  0.5 * np.ones(3) @ self.atoms.get_cell()
            self._commit_atom(atom_type, coords=new_coords)
            return True

        # 2) Bonded placement
        coords = self.choose_vector(atom_type, idx_anchor, limits=limits, max_iter=max_iter)
        if coords is None:
            return False

        #3) slab-based overlap check
        positions = self.atoms.get_positions()
        z_tol = 2.8 # we are checking atoms within 2.8 Angs, so 1-2 layers
        zdiff = np.abs(coords[2] - positions[:, 2])
        close_idxs = np.where(zdiff <= z_tol)[0]
        if not self.check_placement(close_idxs, new_atom_type=atom_type, new_coords=coords):
            return False

        #4) commit to ASE Atoms if OK
        self._commit_atom(atom_type, coords=coords)
        return True


    def slice(self, write_files: Optional = False, raise_by: float = 0) -> None:
        """
        Remove all atoms whose z‐coordinate exceeds the local upper bound,
        then optionally raise that bound everywhere by `raise_by`.

        Parameters
        ----------
        raise_by : float
            Amount to increment the upper‐limit grid (self.limits[3]) by,
            after slicing out the high‐z atoms. Default is 0 (no change).
        write_files: bool
            Write or not files before and after slicing. Default is False.

        Effects
        -------
        - Writes the structure before slicing to "before_slice.xyz" if needed
        - Deletes any atom i for which
              z_i > z_min + self.limits[3][ix,iy] + 1
          where (ix,iy) = find_nearest(x_i, self.limits[0]), find_nearest(y_i, self.limits[1])
          and z_min = min_j z_j.
        - Writes the structure after slicing to "after_slice.xyz" if needed
        - Increments self.limits[3] (the “upper” grid) by `raise_by`.
        """

        if write_files:
            self.atoms.write("before_slice.xyz", format="xyz")

        self.atoms.wrap()
        pos = self.atoms.get_positions()  # (N,3) array
        z_min = pos[:, 2].min() # find the "bottom" of the surface
        up = self.limits[3] # upper limit grid
        # For valid atoms computing z-limits
        z_offset = 1.0

        # Vector indexes XY
        ix = np.floor((pos[:, 0] - self._x0) / self._dx).astype(int)
        iy = np.floor((pos[:, 1] - self._y0) / self._dy).astype(int)
        valid = (ix >= 0) & (ix < self._nx) & (iy >= 0) & (iy < self._ny) #check which atoms we need to cut off

        z_bound = np.full(pos.shape[0], np.inf)
        inds = np.where(valid)[0]
        z_bound[inds] = z_min + up[ix[inds], iy[inds]] + z_offset

        # Mask for atoms that stay
        keep = pos[:, 2] <= z_bound
        n_removed = np.count_nonzero(~keep)
        #print(f"Sliced out {n_removed} atoms")

        # Slicing atoms using mask
        self.atoms = self.atoms[keep]

        self.update_atoms(self.atoms)

        # 2.5) snapshot after slicing
        if write_files:
            self.atoms.write("after_slice.xyz", format = "xyz")

        # 3) raise the upper‐limit grid if requested
        if raise_by:
            # self.limits[3] is a 2D array of upper‐bounds
            self.limits[3] = self.limits[3] + raise_by

    def freeze_bottom_half(self) -> None:
        """
        Fixing the half of atoms on the bottom (down along Z axis) layers of the initial structure,
        Thus adding to self.atoms FixAtoms constraint
        """
        if len(self.atoms) == 0:
            return

        # 1. Checking the Z coords of all atoms
        positions = self.atoms.get_positions()
        z_coords = positions[:, 2]

        # 2. Sorting atoms by their Z coord
        sorted_idx = np.argsort(z_coords)

        # 3. Taking half of bottom atoms
        n = len(sorted_idx)
        bottom_half_idx = sorted_idx[: n // 2].tolist()

        # 4. Saving into self.frozen_indices
        self.frozen_indices = bottom_half_idx.copy()
        self._has_frozen = True

        # 5. Fixing these atoms via ase.constraints.FixAtoms
        constraint = FixAtoms(indices=bottom_half_idx)

        # 6. If there are any other constraint, we need to unite them
        #
        existing = self.atoms.constraints
        if existing:
            self.atoms.set_constraint(existing + [constraint])
        else:
            self.atoms.set_constraint(constraint)