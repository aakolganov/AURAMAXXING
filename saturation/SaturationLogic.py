import numpy as np
from typing import Dict, List, Tuple, Sequence, Optional
from collections import defaultdict
from ase import Atom
from ase.io import write
import logging
from interfaces.MACE_interface import MACEInterface
from numpy.random import Generator
import gc
import torch
from ase.optimize import LBFGS
from default_constants import NORM_COORDINATION, OVER_POS, O_H_BONDLENGTH, El_O_BONDLENGTH
import os


logger = logging.getLogger(__name__)


NO_DOUBLE_SATURATION = True

from base.AmorphousStrucASE import AmorphousStrucASE

class SaturationChildClass(AmorphousStrucASE):
    """
    Represents a specialized child class for handling saturation, extending the functionality of AmorphousStrucASE.

    :ivar energy: The energy associated with the structure.
    :type energy: Optional[float]
    :ivar already_saturated: Status indicating whether the structure is already saturated.
    :type already_saturated: Optional[bool]
    """
    def __init__(self, *args, **kwargs):
        super(SaturationChildClass, self).__init__(*args, **kwargs)
        self.energy = None
        self.already_saturated = None
        self._build_neighbour_list()

    def get_all_cn(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Build a mapping from atom symbol to a dict of coordination numbers and atom indices.
        For fixed atoms, we explicitly state the normal coordination number even if it differs from the actual one.


        Returns:
            cn_map: {
                symbol: {
                    cn_value: [idx1, idx2, ...],
                    ...
                },
                ...
            }
        """
        cn_map: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        # Ensure neighbour list is up to date
        self._build_neighbour_list()
        fixed = self._get_fixed_indices()
        for idx in range(len(self.atoms)):
            sym = self.atoms[idx].symbol
            if idx in fixed:
                cn = NORM_COORDINATION[sym]
            else:
                cn = self.get_cn(idx)
            cn_map[sym][cn].append(idx)
        return cn_map

    def _get_fixed_indices(self) -> set[int]:
        """
        Collects atoms indices that are freezed via ase.constraints.FixAtoms
        """
        fixed = set()
        for constr in self.atoms.constraints:
            fixed.update(constr.index)
        return fixed


    def check_cn(self) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Checks over- and under-coordination for all atoms.
        Returns two dicts:
          over_cn: { symbol: [idx, idx, ...], ... }
          undr_cn: { symbol: [idx, idx, ...], ... }
        Each index repeats by deviation from normal coordination.
        """
        from collections import defaultdict
        over_cn = defaultdict(list)
        undr_cn = defaultdict(list)

        cn_map = self.get_all_cn()
        for sym, cn_dict in cn_map.items():
            if sym not in NORM_COORDINATION:
                raise KeyError(f"No normal coordination for '{sym}'")
            norm = NORM_COORDINATION[sym]
            for cn_val, indices in cn_dict.items():
                delta = cn_val - norm
                if delta > 0:
                    # over-coordinated: repeat indices delta times
                    over_cn[sym].extend(indices * delta)
                elif delta < 0:
                    # under-coordinated: repeat indices -delta times
                    undr_cn[sym].extend(indices * -delta)


        return dict(over_cn), dict(undr_cn)

    def _pbc_dists_to(self, idx_ref: int, idx_list: Sequence[int]) -> np.ndarray:
        """
        Return array of PBC distances from atom idx_ref to each in idx_list.
        """
        ref = self.atoms.get_positions()[idx_ref]
        return np.array([self._pbc_dist(ref, self.atoms.get_positions()[i])
                         for i in idx_list])


    def _pbc_vector(self, idx_from: int, idx_to: int) -> np.ndarray:
        """
        Return the Cartesian minimum‐image displacement vector
        from atom idx_from to atom idx_to.
        """
        pos = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        inv = np.linalg.inv(cell)

        # fractional displacement
        frac_disp = inv.dot(pos[idx_to] - pos[idx_from])
        frac_disp -= np.rint(frac_disp)

        # back to cartesian
        return cell.dot(frac_disp)

    def _make_point(self, idx_ref_atom: int, bond_length: float, tolerance: float) -> np.ndarray:
        """
        Generate a new Cartesian point at distance `bond_length` from a reference atom
        using periodic boundary conditions, then filter out any candidate
        closer than `tolerance` to existing atoms.

        Parameters
        ----------
        idx_ref_atom : int
            Index of the reference atom.
        bond_length : float
            Desired bond length (radius of the sphere).
        tolerance : float
            Minimum allowed distance from any other atom.

        Returns
        -------
        new_point : ndarray (3,)
            Selected Cartesian coordinates for the new atom.
        """
        # 1) Reference position and all positions
        positions = self.atoms.get_positions()
        ref = positions[idx_ref_atom % len(positions)]

        # 2) Identify nearby atoms in a bounding box (to limit cost)
        all_indices = np.arange(len(positions))
        dists_ref = self._pbc_dists_to(idx_ref_atom, all_indices)  # (N,)
        neighbor_mask = (dists_ref <= bond_length + tolerance) & (all_indices != idx_ref_atom)
        neighbor_idxs = all_indices[neighbor_mask]

        # 3) Sample candidate points on the sphere of radius bond_length
        #    Using a coarse mesh in theta/phi
        n_theta = 60
        n_phi = 30
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phis = np.linspace(0, np.pi, n_phi)
        th, ph = np.meshgrid(thetas, phis, indexing='xy')
        x = bond_length * np.sin(ph) * np.cos(th)
        y = bond_length * np.sin(ph) * np.sin(th)
        z = bond_length * np.cos(ph)
        candidates = np.vstack((x.ravel(), y.ravel(), z.ravel())).T + ref

        # 3) Filtering candidates: for each min distance to each neighbour ≥ tolerance
        good = []
        min_dists = []
        for cand in candidates:
            # distances to all nearby-idx
            d2 = np.array([self._pbc_dist(cand, positions[i]) for i in neighbor_idxs])
            if d2.size == 0:
                # if there are no nearby atoms, this is a good candidate.
                return cand
            min_d = d2.min()
            if min_d >= tolerance:
                good.append(cand)
            min_dists.append(min_d)

        if good:
            # return a first "good" candidate
            return good[0]

        # 4) fallback: choose the candidate with the smallest min_dist to any neighbour
        best_idx = int(np.argmax(min_dists))
        return candidates[best_idx]

    def place_new_atom(self, idx_ref_atom: int, new_atom_type: str, bond_length: float, tolerance: float = 1.5) -> None:
        """
        Place a new atom of type `new_atom_type` at a suitable point around
        reference atom `idx_ref_atom`, using bond length and tolerance.

        Parameters
        ----------
        idx_ref_atom : int
            Index of the existing atom to bond to.
        new_atom_type : str
            Chemical symbol of the new atom to insert.
        bond_length : float
            Desired bond length from the reference atom.
        tolerance : float, optional
            Minimum allowed distance to any other atom (default: 1.5 Å).
        """
        # 1) Validate index
        if not (0 <= idx_ref_atom < len(self.atoms)):
            raise IndexError(f"Reference atom index out of range: {idx_ref_atom}")

        # 2) Generate candidate point
        new_point = self._make_point(idx_ref_atom, bond_length, tolerance)
        #wrap it in PBC
        cell = self.atoms.get_cell()
        inv = np.linalg.inv(cell)
        frac = inv.dot(new_point)
        frac -= np.floor(frac)
        new_point = cell.dot(frac)

        # 3) Append atom via ASE and rebuild neighbor list
        atom = Atom(new_atom_type, position=new_point)
        self.atoms.append(atom)
        self.update_atoms(self.atoms)


    def move_atom(self, idx_move: int, idx_ref: int, new_dist: float) -> None:
        """
        Move atom `idx_move` so that its distance to `idx_ref` becomes `new_dist`.
        """
        n = len(self.atoms)
        if not (0 <= idx_move < n and 0 <= idx_ref < n):
            raise IndexError(f"Atom indices out of range: move={idx_move}, ref={idx_ref}")

        # 1) Current vector to take PBCs into account
        disp = self._pbc_vector(idx_ref, idx_move)
        dist0 = np.linalg.norm(disp)
        if dist0 == 0:
            raise ValueError("Atoms coincide; cannot define direction.")

        # 2) New position in Cartesian coordinates
        unit = disp / dist0
        new_cart = self.atoms.get_positions()[idx_ref] + unit * new_dist

        # 3) Wrap into cell
        cell = self.atoms.get_cell()
        inv = np.linalg.inv(cell)
        frac = inv.dot(new_cart)
        frac -= np.floor(frac)
        wrapped = cell.dot(frac)

        # 4) Commit change and update atoms
        self.atoms.positions[idx_move] = wrapped
        self.update_atoms(self.atoms)


    def break_bond(self, idx_atom_move: int, idx_atom_ref: int) -> None:
        """
        Method to break the bond between two atom through rotating the position of one.

        args:
          idx_atom_move (int): Index of the atom that will be moved.
          idx_atom_ref (int): Atom to keep fixed as the reference point
        """
        new_point = self._make_point(idx_atom_ref, El_O_BONDLENGTH, 2.01) # new points
        self.atoms[idx_atom_move].position = new_point #substituting the positions of the atom
        self.update_atoms(self.atoms) # update the atoms

    def check_bound_to_under_coord(self, idx_check_atom: int) -> bool:
        """
        Check if the atom at `idx_check_atom` has any neighbor
        whose actual coordination is below the normal value.

        Parameters
        ----------
        idx_check_atom : int
            Index of the atom to check.

        Returns
        -------
        bool
            True if any neighbor is under-coordinated, False otherwise.
        """
        # 1) Validate index
        n = len(self.atoms)
        if not (0 <= idx_check_atom < n):
            raise IndexError(f"Atom index out of range: {idx_check_atom}")

        # 2) Make sure neighbor list is current
        self._build_neighbour_list()

        # 3) Iterate over neighbors
        nbrs, _ = self._nl.get_neighbors(idx_check_atom)
        for idx_n in nbrs:
            sym = self.atoms[idx_n].symbol
            cn_nb = self.get_cn(idx_n)
            normal = NORM_COORDINATION.get(sym)
            if normal is None:
                raise KeyError(f"No normal coordination defined for '{sym}'")
            if cn_nb < normal:
                return True

        # 4) # If all neighbors are normally coordinated, return False
        return False

    #NOT used in the current implementation

    def over_pos(self, idx_atom_chosen: int) -> None:
        sym = self.atoms[idx_atom_chosen].symbol
        cn_ch = self.get_cn(idx_atom_chosen)
        assert not OVER_POS[sym], f"{sym} should be positively charged, but in OVER_POS[{sym}]=False"
        assert cn_ch > NORM_COORDINATION[sym]

        if self.already_saturated is None:
            self.set_already_saturated()

        # Construct the neighbor list
        self._build_neighbour_list()
        nbrs, _ = self._nl.get_neighbors(idx_atom_chosen)

        # Filtering the "good" neighbours
        def select_good(neighbors):
            good = [
                i for i in neighbors
                if (self.get_cn(i) == NORM_COORDINATION[self.atoms[i].symbol]
                    and i not in self.already_saturated
                    and not self.check_bound_to_under_coord(i))
            ]
            if good:
                return good
            good2 = [
                i for i in neighbors
                if self.get_cn(i) == NORM_COORDINATION[self.atoms[i].symbol]
                   and not self.check_bound_to_under_coord(i)
            ]
            return good2 or list(neighbors)

        best = select_good(nbrs)
        if len(best) > 1:
            d = self._pbc_dists_to(idx_atom_chosen, best)
            idx_sel = best[int(np.argmax(d))]
        else:
            idx_sel = best[0]

        # If cn_ch ≤ 3, fist move, then add O, then H
        if cn_ch <= 3:
            self.move_atom(idx_sel, idx_atom_chosen, new_dist=2.75)
            # ─────────────────────────────────────────────────────────────
            self.place_new_atom(idx_sel, "O", El_O_BONDLENGTH, tolerance=2.01)
            self.place_new_atom(idx_sel, "H", O_H_BONDLENGTH, tolerance=1.5)

        # If (cn_ch > 3) - «break» the bond and first O, then H
        else:
            idx_move = idx_sel
            nbrs2, _ = self._nl.get_neighbors(idx_move)
            candidates = [i for i in nbrs2 if i != idx_atom_chosen]
            idx_ref = self.rng.choice(candidates)
            self.break_bond(idx_move, idx_ref)
            self.place_new_atom(idx_move, "O", El_O_BONDLENGTH, tolerance=2.01)
            self.place_new_atom(idx_move, "H", O_H_BONDLENGTH, tolerance=1.5)


    def over_neg(self, idx_atom_chosen: int) -> None:
        """
        For an under-coordinated (negatively charged) atom at idx_atom_chosen:
          - if its CN ≤ 3: push away its furthest “normal” neighbor and attach an H
          - otherwise: break one bond of a “normal” neighbor and attach an H
        """
        # 1) validation and base data
        n = len(self.atoms)
        if not (0 <= idx_atom_chosen < n):
            raise IndexError(f"Atom index out of range: {idx_atom_chosen}")

        sym = self.atoms[idx_atom_chosen].symbol
        cn_ch = self.get_cn(idx_atom_chosen)
        assert not OVER_POS[sym], f"{sym} must be negatively charged here"
        assert cn_ch > NORM_COORDINATION[sym]

        if self.already_saturated is None:
            self.set_already_saturated()

        # 2) initiate neighbor list
        self._build_neighbour_list()
        nbrs, _ = self._nl.get_neighbors(idx_atom_chosen)

        # 3) Auxillary set of «good» neighbors based on the coordiation
        def select_good(neighbors):
            # neihgbors which have CN == "normal" aren't connected to undercoordinated atoms.
            good = [
                i for i in neighbors
                if (self.get_cn(i) == NORM_COORDINATION[self.atoms[i].symbol]
                    and i not in self.already_saturated
                    and not self.check_bound_to_under_coord(i))
            ]
            if good:
                return good
            # less strict criterion: only CN == "normal"
            good2 = [
                i for i in neighbors
                if self.get_cn(i) == NORM_COORDINATION[self.atoms[i].symbol]
                   and not self.check_bound_to_under_coord(i)
            ]
            return good2 or list(neighbors)

        # 4) Finding  «best» neighbors
        best = select_good(nbrs)
        if len(best) > 1:
            # the furthest away
            d = self._pbc_dists_to(idx_atom_chosen, best)
            idx_sel = best[int(np.argmax(d))]
        else:
            idx_sel = best[0]

        # 5) If CN ≤ 3
        if cn_ch <= 3:
            # 5a) Moving the neighbor
            self.move_atom(idx_sel, idx_atom_chosen, new_dist=2.75)
            # 5b) attach H
            self.place_new_atom(idx_sel, "H", O_H_BONDLENGTH, tolerance=1.5)

        # 6) If CN > 3
        else:
            # 6a) Selecting «moving» atom
            idx_move = idx_sel
            # 6b) Selecting the random from its neighbors
            nbrs2, _ = self._nl.get_neighbors(idx_move)
            candidates = [i for i in nbrs2 if i != idx_atom_chosen]
            idx_ref = self.rng.choice(candidates)

            # 6c)«breaking the bond и attach H
            self.break_bond(idx_move, idx_ref)
            self.place_new_atom(idx_move, "H", O_H_BONDLENGTH, tolerance=1.5)

    def undr_pos(self, idx_atom_chosen: int) -> None:
        """
        Method to saturate an under-coordinated atom which has a positive formal charge with a -OH fragment

        args:
          idx_atom_chosen (int): atom to bind the -OH fragment to

        raises:
          AssertError: If the atom indexed is not under-coordinated
          AssertError: If the atom indexed is not considered negatively charged when under-coordinated
        """
        chosen_atom = self.atoms[idx_atom_chosen]
        sym = chosen_atom.symbol
        chosen_atom_cn = self.get_cn(idx_atom_chosen)
        assert not OVER_POS[sym] is True, f"{sym} is negativly charged when under-coordinated"
        assert chosen_atom_cn < NORM_COORDINATION[sym]

        self.place_new_atom(idx_atom_chosen, "O", El_O_BONDLENGTH, tolerance=2.0)
        idx_O = len(self.atoms)-1
        self.place_new_atom(idx_O, "H", O_H_BONDLENGTH, tolerance=1.5)

    def undr_neg(self, idx_atom_chosen: int) -> None:
        """
        Method to saturate an under-coordinated atom which has a negative formal charge with a -H fragment

        args:
          idx_atom_chosen (int): atom to bind the -H fragment to

        raises:
          AssertError: If the atom indexed is not under-coordinated
          AssertError: If the atom indexed is not considered positively charged when under-coordinated
        """
        chosen_atom = self.atoms[idx_atom_chosen]
        sym = chosen_atom.symbol
        chosen_atom_cn = self.get_cn(idx_atom_chosen)
        assert not OVER_POS[sym] is False, f"{sym} is positivly charged when under-coordinated"
        assert chosen_atom_cn < NORM_COORDINATION[sym]

        self.place_new_atom(idx_atom_chosen, "H", O_H_BONDLENGTH, tolerance=1.5)

    def set_already_saturated(self) -> None:
        """
        Mark as 'saturated' any atom that is the 1st or 2nd neighbor
        of any hydrogen atom, to avoid over-fragmentation around H.
        """
        # 1) Update neighbor list if needed
        self._build_neighbour_list()

        # 2) All H-аtoms
        H_idxs = [i for i, atom in enumerate(self.atoms) if atom.symbol == "H"]

        sat_set = set()
        for h in H_idxs:
            # 1 coord sphere
            nbrs1, _ = self._nl.get_neighbors(h)
            for n1 in nbrs1:
                sat_set.add(n1)
                # second coord sphere
                nbrs2, _ = self._nl.get_neighbors(n1)
                sat_set.update(nbrs2)

        # 3) Delete H from sat_set
        sat_set -=set(H_idxs)
        self.already_saturated = np.array(
            [i for i in sat_set if self.atoms[i].symbol != "H"],
            dtype=int)

    # def ring_paths(self, size: int) -> np.ndarray:
    #     """
    #     Find all the cycles of length size in the structure.
    #     """
    #     self._build_neighbour_list()
    #     G = nx.Graph()
    #     n = len(self.atoms)
    #     G.add_nodes_from(range(n))
    #     for i in range(n):
    #         nbrs, _ = self._nl.get_neighbors(i)
    #         for j in nbrs:
    #             if j > i:
    #                 G.add_edge(i, j)
    #
    #     # Transform into directed graph and find all_simple_cycles
    #     D = nx.DiGraph(G)
    #     cycles = []
    #     for cycle in nx.simple_cycles(D):
    #         if len(cycle) == size:
    #             # sort the cycl
    #             cycles.append(tuple(sorted(cycle)))
    #
    #     # deleting the duplicates
    #     unique = list({c for c in cycles})
    #     return np.array(unique, dtype=int)
    #
    # def idx_in_rings(self, size: int = None, paths = None):
    #     if size is None and paths is None:
    #         raise AssertionError("Need to choose one option. Either set the size of the ring or provide paths.")
    #
    #     if not size is None:
    #         paths = self.ring_paths(size)
    #     if not paths is None:
    #         paths = paths
    #     syms = np.array(self.atoms.get_chemical_symbols())
    #     flattened = paths.flatten()
    #     all_Si_in_ring = flattened[syms[flattened] == "Si"]
    #     all_O_in_ring = flattened[syms[flattened] == "O"]
    #     return all_Si_in_ring, all_O_in_ring


class Generation:
    """Manage one generation: a pool of parents and the children produced from them."""
    num_generation = 0

    def __init__(self, max_children: int,
                 rng: Optional[Generator] = None,
                 seed: Optional[int] = None,
                 mace_model_path: str = "",
                 device: str = "mps"):

        self.max_children = max_children
        self.parents: List[SaturationChildClass] = []
        self.children: List[SaturationChildClass] = []
        self.rng = np.random.default_rng()
        Generation.num_generation += 1
        # Initializing local generator:
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        elif isinstance(seed, int):
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        self.mace_model_path = mace_model_path
        self.device = device

    def add_parent(self, parent: SaturationChildClass) -> None:
        """Add a parent to the pool."""
        if not isinstance(parent, SaturationChildClass):
            raise TypeError(f"Expected SaturationChildClass, got {type(parent)}")
        self.parents.append(parent)


    def choose_parent(self) -> SaturationChildClass:
        """Randomly pick one parent weighted by exp(-energy)."""
        if not self.parents:
            raise RuntimeError("No parents to choose from.")
        energies = [p.energy if p.energy is not None else float('inf')
                    for p in self.parents]
        # If there is one parent
        if len(self.parents) == 1:
            return self.parents[0]
        weights = np.exp(-np.array(energies, float))
        # inf energies → weight = 0
        weights /= weights.sum()
        idx = self.parents[0].rng.choice(len(self.parents), p=weights)
        return self.parents[idx]

    def add_child(self, child: SaturationChildClass) -> None:
        """Add a child to the pool (up to max_children)."""
        if len(self.children) >= self.max_children:
            raise RuntimeError("Reached maximum number of children.")
        self.children.append(child)

    def new_parents(self, k: int) -> List[SaturationChildClass]:
        """Select k children with lowest energy to become next parents."""
        # Treat None-energy as +inf so they never get picked
        sorted_children = sorted(
            self.children,
            key=lambda c: c.energy if c.energy is not None else float('inf')
        )
        return sorted_children[:k]

    def dump_children(self) -> None:
        """
        For each generation, create a dedicated folder named "generation_<N>".
        Inside that folder:
          • all_children_gen<N>.xyz   → a multi‐frame XYZ with every child, with energy in the comment line
          • child_<i>.cif             → a separate CIF file for each child (indexed from 1)
        """
        gen_dir = f"generation_{Generation.num_generation}"
        os.makedirs(gen_dir, exist_ok=True)

        # Path for the multi‐frame XYZ
        xyz_filename = f"all_children_gen{Generation.num_generation}.xyz"
        xyz_path = os.path.join(gen_dir, xyz_filename)

        # If a previous XYZ exists in this folder, delete it
        if os.path.exists(xyz_path):
            os.remove(xyz_path)

        # Loop over children, append each one to the multi‐frame XYZ
        for idx, child in enumerate(self.children, start=1):
            energy = child.energy if child.energy is not None else "not optimized"
            # Store the energy in the ASE‐info so that ASE writes it as the comment for this frame
            child.atoms.info['comment'] = str(energy)

            # Append to the multi‐frame XYZ
            write(xyz_path, child.atoms, format='xyz', append=True)

            # Also write out a separate CIF for this child
            cif_name = f"child_{idx}.cif"
            cif_path = os.path.join(gen_dir, cif_name)
            write(cif_path, child.atoms, format='cif')

    def make_gen_sat(self, max_steps: int=100):
        """
        Method of making a new generation of structures. Will make children until it had made as many as defined as the
        maximum number of children per generation.
        Chooses a parent, determines if a under- or over-coordinated atom will be used to fulfill saturation. Under-coordinated
        atoms get priority. Then chooses the indicies of the atoms that will be used for saturation and add H2O according to
        internal logic. Finishes with optimizing the geometry of the child.

        args:
          max_steps (int): maximum number of geometry optmization steps
        """

        mace_if = MACEInterface(self.mace_model_path, device=self.device)

        i=1 #count of children

        n_children = len(self.children)
        print(f"Starting optimizing {n_children} child(ren) in generation {Generation.num_generation}", flush=True)


        for _ in range(self.max_children):


            # 1) selecting parent and cloning it
            parent: SaturationChildClass = self.choose_parent()
            new_child = SaturationChildClass(
                atoms=parent.atoms.copy(),
                cell=parent.atoms.get_cell(),
                pbc=parent.atoms.get_pbc(),
                seed=parent.rng
            )


            # 2) cope the rest of parameters (cutoffs, already_saturated и т.д.)
            new_child.cut_offs = parent.cut_offs.copy()
            new_child.max_cn = parent.max_cn.copy()
            new_child.already_saturated = (
                parent.already_saturated.copy() if parent.already_saturated is not None else []
            if parent.already_saturated is not None else []
                            )

            # 3) check over/undercoordination
            over_cn, undr_cn = new_child.check_cn()

            # 4) # find if there are remaining atoms of negative and positive charge which are under coordinated
            remaining_undr_pos = sum(
                len(idxs) for atype, idxs in undr_cn.items()
                if not OVER_POS[atype]
            )
            remaining_undr_neg = sum(
                len(idxs) for atype, idxs in undr_cn.items()
                if OVER_POS[atype]
            )

            pos_atom = "undr" if remaining_undr_pos > 0 else "over"
            neg_atom = "undr" if remaining_undr_neg > 0 else "over"

            # 5) select atoms
            pos_idx, neg_idx = self.choose_atoms(
                pos_atom, neg_atom,
                over_cn, undr_cn,
                new_child.already_saturated
            )

            # 6) saturating
            self.saturate(new_child, pos_idx, pos_atom, neg_idx, neg_atom)
            new_child.set_already_saturated()

            # 6.5) final wrap ups - not needed, all the update in other functions
            #new_child.update_atoms(atoms=new_child.atoms)

            #7) Optimizing generated chidren

            print (f"Optimizing children {i}/{range(self.max_children)}")


            steps = int(np.ceil(0.5 * max_steps)) if (pos_atom == "undr" and neg_atom == "undr") else max_steps
            energy_child, optimized_atoms = mace_if.optimize(new_child.atoms, max_steps=steps)
            new_child.atoms = optimized_atoms
            new_child.energy = energy_child

            i+=1

            # 8) adding children to a list and dump them to the storage
            self.add_child(new_child)
            self.dump_children()




            print(f"Finished generation {Generation.num_generation}")

    _SATURATE_DISPATCH = {
        ("undr", "undr"): ("undr_pos", "undr_neg"),
        ("undr", "over"): ("undr_pos", "over_neg"),
        ("over", "undr"): ("over_pos", "undr_neg"),
    }

    def saturate(self,
                 child: SaturationChildClass,
                 pos_atom_idx: int, pos_state: str,
                 neg_atom_idx: int, neg_state: str) -> None:
        """
        Apply the correct saturation routines on the 'positive' and 'negative' atoms
        of a child structure, based on whether they are under- or over-coordinated.

        :param child: the new child structure to modify
        :param pos_atom_idx: index of the ‘positive’ atom in child.atoms
        :param pos_state: 'undr' or 'over'
        :param neg_atom_idx: index of the ‘negative’ atom
        :param neg_state: 'undr' or 'over'
        :raises ValueError: if both states are 'over' or if unknown states provided
        """
        key = (pos_state, neg_state)
        if key not in self._SATURATE_DISPATCH:
            raise ValueError(f"Cannot saturate with states {key!r} (only {list(self._SATURATE_DISPATCH)})")

        # Log once with info
        logger.info("Saturating child: pos_atom[%d]=%s, neg_atom[%d]=%s",
                    pos_atom_idx, pos_state, neg_atom_idx, neg_state)

        # unpacking to see which methods we want to invoke
        pos_method, neg_method = self._SATURATE_DISPATCH[key]

        for method_name, idx in [(pos_method, pos_atom_idx), (neg_method, neg_atom_idx)]:
            if not hasattr(child, method_name):
                raise AttributeError(f"Child has no method {method_name!r}")
            # calling, transfering the index
            getattr(child, method_name)(idx)


    @staticmethod
    def split_pos_neg(over_cn: dict, undr_cn: dict) -> tuple[list[int], list[int]]:
        """
        Static method to splits the over- and under-coordinated atom distionaries into whether they are considered positively or
        negatively charged.

        args:
          over_cn: dictionary of the indicies of the over-coordinated atoms.
          undr_cn: dictionary of the indicies of the under-coordinated atoms.

        returns:
          Two list, one of the indicies of all positive atoms and of the indicies of all negative atoms present.
        """
        pos_atoms: List[int] = []
        neg_atoms: List[int] = []

        # First approach: overcoord dict → calculating OVER_POS==True as positive
        # Second approach: for undercoord dict → calculating OVER_POS==False as negative
        for cn_dict, positive_when_over in ((over_cn, True), (undr_cn, False)):
            for atom_type, indices in cn_dict.items():
                # selecting to which list to add
                target = pos_atoms if OVER_POS[atom_type] == positive_when_over else neg_atoms
                target.extend(indices)

        return pos_atoms, neg_atoms

    def choose_atoms(
        self,
        pos_atom: str,
        neg_atom: str,
        over_cn: Dict[str, List[int]],
        undr_cn: Dict[str, List[int]],
        already_saturated: List[int]
    ) -> Tuple[int, int]:
        """
        Static method to choose the specific index of the atom to saturate depending on if we will saturate an over- or under-coordinated atoms
        for the 'formally positive' and 'formally negative' atoms. If the flag for no doubles `NO_DOUBLE_SATURATION` is set to True it will
        filter out the indicies which have already been saturated once from the list.

        args:
          pos_atom (str): flag for it the positive atom will have to be over- or under-coordinated. Should be either 'undr' or 'over'
          neg_atom (str): flag for it the negative atom will have to be over- or under-coordinated. Should be either 'undr' or 'over'
          over_cn (Dict[str, List[int]]): Dictionary of the indices of all over-coordinated atoms within the structure organized as
                                          key-atom type, value-list of indicies of over-coordinated atoms of that type.
          undr_cn (Dict[str, List[int]]): Dictionary of the indices of all under-coordinated atoms within the structure organized as
                                          key-atom type, value-list of indicies of under-coordinated atoms of that type.
          already_saturated (List[int]): List of all the indicies of atoms which have already been saturated.

        returns:
          A tuple of the index of the positive and negative atoms

        raises:
          ValueError: if pos_atom or neg_atom have an incorrect string for the match function.
        """
        def flatten(cn_dict: Dict[str, List[int]], *,
                    want_over: bool) -> List[int]:
            """
            Collect from cn_dict (over_cn ore undr_cn) all indeces those types,
            for which OVER_POS[type] == want_over.
            """
            out = []
            for atom_type, idxs in cn_dict.items():
                if OVER_POS[atom_type] == want_over:
                    out.extend(idxs)
            return out

        def pick(idxs: List[int]) -> int:
            if not idxs:
                raise ValueError("There are no available atoms")
            return int(self.rng.choice(idxs))


        def filter_double(idxs: List[int]) -> List[int]:
            """ If needed - remove the saturated atoms from the list of available atoms."""
            if not NO_DOUBLE_SATURATION:
                return idxs
            filtered = [i for i in idxs if i not in already_saturated]
            return filtered or idxs  # if need -> go back to the original

        # 1) Positively charged atom
        if pos_atom == "undr":
            pos_list = flatten(undr_cn, want_over=False)
        elif pos_atom == "over":
            pos_list = flatten(over_cn, want_over=True)
            pos_list = filter_double(pos_list)
        else:
            raise ValueError(f"Unknown pos_atom flag: {pos_atom!r}")

        pos_atom_idx = pick(pos_list)

        # 2) Negatively charged atom
        if neg_atom == "undr":
            neg_list = flatten(undr_cn, want_over=True)
        elif neg_atom == "over":
            neg_list = flatten(over_cn, want_over=False)
            neg_list = filter_double(neg_list)
        else:
            raise ValueError(f"Unknown neg_atom flag: {neg_atom!r}")

        neg_atom_idx = pick(neg_list)

        return pos_atom_idx, neg_atom_idx

    def opt_children(self, max_steps: int) -> None:
        """
        Optimizes all children in this generation using MACE..

        Parameters
        ----------
        max_steps : int
            max iterations.
        mace_model_path : str
            Path to the MACE model file..
        device : str, optional
            Device to calculate.
        """
        # 1) Initialization MACEInterface:
        mace_if = MACEInterface(self.mace_model_path, device=self.device)

        child_n = len(self.children)

        # 2) Going through all children and optimizing them:
        for idx, child in enumerate(self.children, start=1):
            print(f"Optimizing child {idx}/{child_n} with MACE…", flush=True)

            # a) Take Atoms from child:
            atoms = child.atoms  #

            # b) Setting the calculator

            atoms.set_calculator(mace_if)

            # c) Optimizing w/ LBFGS

            opt = LBFGS(atoms)
            opt.run(fmax=0.1, steps=max_steps)


            # d) Taking the energy:

            try:
                energy = atoms.get_potential_energy()
            except Exception:
                energy = float("nan")

            # Updating child data
            child.atoms = atoms
            child.energy = energy

            #e) Unlink the calculator
            atoms.set_calculator(None)
            del opt

            #f) Initiating garbage collector to clean the MPS cash

            gc.collect()
            if mace_if.device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        #4) Announce the results
        print(f"finished optimizing {child_n} children.", flush=True)






