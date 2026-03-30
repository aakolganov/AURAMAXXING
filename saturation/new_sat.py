from base.amorphous_structure import AmorphousStruc
from interfaces.base_interface import CalculatorInterface
from typing import Optional
import numpy as np
from helpers.files_io import highlight_coordination
from ase.geometry import find_mic
from default_constants import OXIDATION_POS, OVER_POS, d_min_max
from helpers.atom_placing import place_atom_sphere, place_atom_force
    
def move_atom(
    amorph_struct: AmorphousStruc,
    idx_move: int,
    move_away_from: int,
    dist_move: float,
    iterations: int = 10,
    alpha: float = 0.5,
):
    """
    Moves idx_move away from move_away_from to a target distance, then
    iteratively relaxes the local environment to reduce strain.

    This function first applies a direct displacement to `idx_move`. Then, it
    runs a number of iterations where the neighbors of `idx_move` (and `idx_move`
    itself) are adjusted to partially restore their original bond lengths,
    simulating a local spring-based relaxation.

    Args:
        amorph_struct: The amorphous structure object.
        idx_move: Index of the atom to move.
        move_away_from: Index of the atom to move away from.
        dist_move: The target distance between idx_move and move_away_from.
        iterations: Number of relaxation iterations.
        alpha: Damping factor for position updates (0 < alpha <= 1). A smaller
               value leads to more gentle, stable relaxation.
    """
    pos = amorph_struct.atoms.get_positions()
    cell = amorph_struct.atoms.get_cell()
    pbc = amorph_struct.atoms.get_pbc()
    graph = amorph_struct.get_graph(force_rebuild=True)

    # 1. Store original distances to neighbors of idx_move
    neighbors = [n for n in graph.neighbors(idx_move) if n != move_away_from]
    original_distances = {}
    for n_idx in neighbors:
        dist = amorph_struct.atoms.get_distance(idx_move, n_idx, mic=True)
        original_distances[n_idx] = dist

    # 2. Apply the initial primary displacement to idx_move
    vec, dist = find_mic(pos[idx_move] - pos[move_away_from], cell, pbc)

    if dist < 1e-6:
        # If atoms are on top of each other, move in a random direction.
        vec = amorph_struct.rng.normal(size=3)
        dist = np.linalg.norm(vec)

    if dist > 1e-6:
        disp = (vec / dist) * (dist_move - dist)
        pos[idx_move] += disp

    # 3. Iteratively relax the local structure
    for _ in range(iterations):
        displacements = np.zeros_like(pos)
        for n_idx in neighbors:
            vec_mn, dist_mn = find_mic(pos[n_idx] - pos[idx_move], cell, pbc)
            if dist_mn < 1e-6: continue
            error = dist_mn - original_distances[n_idx]
            disp_n = (-vec_mn / dist_mn) * error * alpha
            displacements[n_idx] += disp_n
            displacements[idx_move] -= disp_n
        pos += displacements

    indices_to_update = [idx_move] + neighbors
    symbols = amorph_struct.symbols

    for idx in indices_to_update:
        amorph_struct.replace_atom(symbols[idx], pos[idx], idx)
    amorph_struct.atoms.wrap()


def collect_over_or_under_cn_atoms(amorphous_struct: AmorphousStruc, do_under: bool):
    all_cn = amorphous_struct.get_cn()

    symbols = np.array(amorphous_struct.symbols)
    cn_dict = {at: [] for at in set(symbols)}
    for sym, limit in amorphous_struct.min_cn.items():
        if sym not in symbols:
            continue
        
        mask = (symbols == sym)
        mask &= (all_cn < limit) if do_under else (all_cn > limit)
        
        cn_dict[sym].extend(np.where(mask)[0])
    return cn_dict


def select_idx_for_move(amorphous_struct: AmorphousStruc, idx_selection: list[int]) -> tuple[int, int]:
    chosen_idx = amorphous_struct.rng.choice(idx_selection)
    graph = amorphous_struct.get_graph()
    neighbors = list(graph.neighbors(chosen_idx))
    dists = [amorphous_struct.atoms.get_distance(chosen_idx, n, mic=True) for n in neighbors]
    idx_furthest = neighbors[np.argmax(dists)]

    if amorphous_struct.get_cn(chosen_idx) < amorphous_struct.get_cn(idx_furthest):
        return chosen_idx, idx_furthest
    return idx_furthest, chosen_idx


def find_tetrogonal_sites(amorphous_struct: AmorphousStruc) -> list[int]:
    """
    Identifies atoms that have coordination number strictly greater than their minimum
    but less than or equal to their maximum coordination number.
    """
    variable_cn_types = [
        sym for sym, min_c in amorphous_struct.min_cn.items()
        if min_c != amorphous_struct.max_cn.get(sym, min_c)
    ]
    
    if not variable_cn_types:
        return []

    all_cn = amorphous_struct.get_cn()
    symbols = np.array(amorphous_struct.symbols)
    found_indices = []

    for sym in variable_cn_types:
        min_c = amorphous_struct.min_cn[sym]
        max_c = amorphous_struct.max_cn[sym]
        mask = (symbols == sym) & (all_cn > min_c) & (all_cn <= max_c)
        found_indices.extend(np.where(mask)[0].tolist())
        
    return found_indices


def saturate_under_coordinated(
        amorphous_struct: AmorphousStruc,
        bond_lengths = {"O": 1.2, "H": 0.96},
    ):
    """ Does the basic saturation of atoms through adding OH to positively charged and H to negatively charged. Does not Optimize structure."""
    amorphous_struct.atoms.wrap()

    highlight_coordination(amorphous_struct, "highlighted_initial.xyz")
    undr_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=True)


    def try_then_force_place(place_atom: str, attach_idx: int):
        is_placed = place_atom_sphere(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx, num_samples=250, bond_length=bond_lengths[place_atom])
        if not is_placed:
            is_placed = place_atom_force(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx, num_samples=250, bond_length=bond_lengths[place_atom])

    for sym, idx_list in undr_cn.items():
        saturate_with_OH = sym in OXIDATION_POS
        for attach_idx in idx_list:
            if saturate_with_OH:
                try_then_force_place("O", attach_idx)
                attach_idx = len(amorphous_struct) - 1 # to account for the 0 index
            print(amorphous_struct.atoms[attach_idx].symbol)
            try_then_force_place("H", attach_idx)


def correct_charge(
        amorphous_struct: AmorphousStruc,
        bond_lengths = {"O": 1.2, "H": 0.96},
    ):
    """ Creates a charge neutral surface through adding H and OH until correct. Add to over-coordinated atoms. Does not Optimize structure. """
    def try_then_force_place(place_atom: str, attach_idx: int):
        is_placed = place_atom_sphere(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx, num_samples=250, bond_length=bond_lengths[place_atom])
        if not is_placed:
            is_placed = place_atom_force(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx, num_samples=250, bond_length=bond_lengths[place_atom])

    current_charge = amorphous_struct.charge()
    while current_charge != 0:
        if current_charge > 0:
            # implied positive charge so move an over-coordinated atom which is positively charged
            over_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=False)
            indices = [i for k, v in OVER_POS.items() if v and k in over_cn for i in over_cn[k]]
        else:
            undr_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=True)
            indices = [i for k, v in OVER_POS.items() if not v and k in undr_cn for i in undr_cn[k]]
        
        if len(indices) == 0:
            indices = find_tetrogonal_sites(amorphous_struct)

        chosen_idx_pos, idx_furthest = select_idx_for_move(amorphous_struct, indices)
        move_atom(
            amorphous_struct, 
            idx_move=chosen_idx_pos, 
            move_away_from=idx_furthest, 
            dist_move=d_min_max[amorphous_struct.atoms[chosen_idx_pos].symbol][amorphous_struct.atoms[idx_furthest].symbol][0]+0.2
            )
        
        attach_idx = idx_furthest
        if current_charge > 0:
            try_then_force_place("O", attach_idx)
            attach_idx = len(amorphous_struct) - 1 
        try_then_force_place("H", attach_idx)
        current_charge = amorphous_struct.charge()

    amorphous_struct.sort_atoms()
    amorphous_struct.atoms.write("before_opt.vasp", format="vasp")

    