from typing import Optional

from base.amorphous_structure import AmorphousStruc
import numpy as np
from helpers.files_io import highlight_coordination
from ase.geometry import find_mic
from default_constants import (
    OXIDATION_POS,
    OVER_POS,
    d_min_max,
    El_O_BONDLENGTH,
    O_H_BONDLENGTH,
)
from helpers.atom_placing import place_atom_sphere, place_atom_force, place_atom_terminal

# Bond lengths used when attaching saturation groups: O onto an under-coordinated
# Si/Al (El_O_BONDLENGTH ~ 1.63 A) and H onto an O (O_H_BONDLENGTH ~ 0.98 A).
DEFAULT_SAT_BOND_LENGTHS = {"O": El_O_BONDLENGTH, "H": O_H_BONDLENGTH}


def _try_then_force_place(amorphous_struct, place_atom: str, attach_idx: int, *,
                          num_samples: int, bond_lengths: dict) -> None:
    """Attach ``place_atom`` to ``attach_idx`` at its saturation bond length: try the
    collision-aware spherical placement first, and fall back to the always-succeeds
    least-overlap placement if that is sterically blocked, so a cap is always added."""
    if not place_atom_sphere(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx,
                             num_samples=num_samples, bond_length=bond_lengths[place_atom]):
        place_atom_force(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx,
                         num_samples=num_samples, bond_length=bond_lengths[place_atom])


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
            if dist_mn < 1e-6:
                continue
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
    # under-coordinated: CN below the (per-element) minimum; over-coordinated: CN above
    # the per-atom maximum (so an Al allowed CN 6 isn't flagged over until CN > 6).
    limits = amorphous_struct.min_cn_array() if do_under else amorphous_struct.max_cn_array()
    flagged = (all_cn < limits) if do_under else (all_cn > limits)
    for sym in set(symbols):
        mask = (symbols == sym) & flagged
        cn_dict[sym].extend(np.where(mask)[0])
    return cn_dict


def select_idx_for_move(amorphous_struct: AmorphousStruc,
                        idx_selection: list[int]) -> Optional[tuple[int, int]]:
    """Pick an (atom_to_move, pivot_neighbour) pair from ``idx_selection``. The move pivots on
    the chosen atom's furthest-bonded neighbour, so only atoms that actually have a neighbour
    are eligible. Returns ``None`` when none of the candidates is bonded to anything (e.g. an
    isolated under-coordinated atom), so the caller can stop gracefully instead of taking
    ``argmax`` of an empty neighbour list."""
    graph = amorphous_struct.get_graph()
    movable = [i for i in idx_selection if graph.degree(i) > 0]
    if not movable:
        return None
    chosen_idx = amorphous_struct.rng.choice(movable)
    neighbors = list(graph.neighbors(chosen_idx))
    dists = [amorphous_struct.atoms.get_distance(chosen_idx, n, mic=True) for n in neighbors]
    idx_furthest = neighbors[int(np.argmax(dists))]

    if amorphous_struct.get_cn(chosen_idx) < amorphous_struct.get_cn(idx_furthest):
        return chosen_idx, idx_furthest
    return idx_furthest, chosen_idx


def _prune_orphans_from_move(amorphous_struct, cn_before, n_before: int) -> int:
    """Remove atoms that the move in this iteration orphaned and the re-cap did not fix.

    ``correct_charge`` pushes an atom a bond-length away from a neighbour to break a bond; if
    that neighbour had no other bonds it is left isolated (CN 0). The loop then tries to re-cap
    it with an H/OH, but when that placement fails the atom stays dangling and would later
    crash move selection. Here we drop exactly those: atoms present before the move (index <
    ``n_before``), not fixed, that were bonded before but are isolated now. Atoms placed by the
    re-cap (index >= ``n_before``) and pre-existing isolated atoms are left untouched."""
    amorphous_struct.get_graph(force_rebuild=True)
    cn_now = amorphous_struct.get_cn()
    fixed = amorphous_struct.fixed_indices()
    orphans = [i for i in range(n_before)
               if i not in fixed and cn_before[i] > 0 and cn_now[i] == 0]
    if not orphans:
        return 0
    mask = np.zeros(len(amorphous_struct), dtype=bool)
    mask[orphans] = True
    amorphous_struct.remove_atom(mask)
    return len(orphans)


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
    # Per-atom bounds so a tagged Al (max 6) counts as variable-CN up to CN 6.
    min_arr = amorphous_struct.min_cn_array()
    max_arr = amorphous_struct.max_cn_array()
    found_indices = []

    for sym in variable_cn_types:
        mask = (symbols == sym) & (all_cn > min_arr) & (all_cn <= max_arr)
        found_indices.extend(np.where(mask)[0].tolist())

    return found_indices


def saturate_under_coordinated(
        amorphous_struct: AmorphousStruc,
        bond_lengths=None,
        num_samples: int = 250,
        highlight_file: Optional[str] = None,
    ):
    """ Does the basic saturation of atoms through adding OH to positively charged and H to negatively charged. Does not Optimize structure."""
    if bond_lengths is None:
        bond_lengths = DEFAULT_SAT_BOND_LENGTHS
    amorphous_struct.atoms.wrap()

    # Optional debug dump highlighting coordination defects; off by default so production
    # runs don't write a file into the working directory.
    if highlight_file is not None:
        highlight_coordination(amorphous_struct, highlight_file)
    undr_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=True)

    for sym, idx_list in undr_cn.items():
        saturate_with_OH = sym in OXIDATION_POS
        for attach_idx in idx_list:
            if saturate_with_OH:
                _try_then_force_place(amorphous_struct, "O", attach_idx,
                                      num_samples=num_samples, bond_lengths=bond_lengths)
                attach_idx = len(amorphous_struct) - 1 # to account for the 0 index
            _try_then_force_place(amorphous_struct, "H", attach_idx,
                                  num_samples=num_samples, bond_lengths=bond_lengths)


def correct_charge(
        amorphous_struct: AmorphousStruc,
        bond_lengths=None,
        max_iterations: int = 1000,
        num_samples: int = 250,
        move_alpha: float = 0.5,
    ):
    """ Creates a charge neutral surface through adding H and OH until correct. Add to over-coordinated atoms. Does not Optimize structure.

    Each iteration breaks one bond and re-caps with an H/OH group, which should change
    the net formal charge by one unit toward zero. It is not guaranteed to, though:
    ``_prune_orphans_from_move`` can delete a charged atom the move orphaned (e.g. a -2 O
    or a +1 H), so a single iteration can jump the charge by more than one and even
    *overshoot* past zero -- after which there may be no candidate left to come back, and
    the slab would ship charged. To stay robust we therefore *only commit an iteration that
    strictly reduces ``abs(charge)``*: each attempt is taken on a snapshot and reverted if
    it overshoots or makes no progress, and the next attempt re-draws a different
    candidate (the rng has advanced). The loop can thus never cross zero. ``max_iterations``
    caps the total attempts; ``max_stalls`` caps consecutive reverts so an unfixable slab
    stops instead of spinning. If it still cannot reach neutrality, a clear error is raised
    rather than silently returning a charged slab.
    """
    if bond_lengths is None:
        bond_lengths = DEFAULT_SAT_BOND_LENGTHS

    current_charge = amorphous_struct.charge()
    iteration = 0
    stalls = 0
    max_stalls = max(20, 4 * len(amorphous_struct))
    while current_charge != 0:
        if iteration >= max_iterations or stalls >= max_stalls:
            break
        iteration += 1
        if current_charge > 0:
            # implied positive charge so move an over-coordinated atom which is positively charged
            over_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=False)
            indices = [i for k, v in OVER_POS.items() if v and k in over_cn for i in over_cn[k]]
        else:
            undr_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=True)
            indices = [i for k, v in OVER_POS.items() if not v and k in undr_cn for i in undr_cn[k]]

        if len(indices) == 0:
            indices = find_tetrogonal_sites(amorphous_struct)
        if len(indices) == 0:
            break

        move = select_idx_for_move(amorphous_struct, indices)
        if move is None:
            break
        chosen_idx_pos, idx_furthest = move

        # Snapshot the atoms so an overshooting attempt can be reverted. We do NOT snapshot
        # the rng, so the retry after a revert re-draws a different candidate/placement.
        snapshot = amorphous_struct.atoms.copy()
        cn_before = amorphous_struct.get_cn()
        n_before = len(amorphous_struct)
        move_atom(
            amorphous_struct,
            idx_move=chosen_idx_pos,
            move_away_from=idx_furthest,
            dist_move=d_min_max[amorphous_struct.atoms[chosen_idx_pos].symbol][amorphous_struct.atoms[idx_furthest].symbol][0]+0.2,
            alpha=move_alpha,
            )

        # Cap with a bystander-aware placement: the cap bonds only its intended attach atom
        # where possible, so a forced placement can't over-coordinate unrelated atoms (M2).
        # (The attach atom itself may still gain a bond beyond its max -- the move above was
        # meant to free a slot but is left as a no-op, so this only guards the bystanders.)
        attach_idx = idx_furthest
        if current_charge > 0:
            place_atom_terminal(amorphous_struct, "O", attach_idx,
                                bond_length=bond_lengths["O"], num_samples=num_samples)
            attach_idx = len(amorphous_struct) - 1
        place_atom_terminal(amorphous_struct, "H", attach_idx,
                            bond_length=bond_lengths["H"], num_samples=num_samples)
        # Drop any atom the move orphaned that the re-cap above failed to bond, so it can't
        # dangle in the output or crash a later iteration's move selection.
        _prune_orphans_from_move(amorphous_struct, cn_before, n_before)

        new_charge = amorphous_struct.charge()
        if abs(new_charge) < abs(current_charge):
            current_charge = new_charge          # genuine progress toward zero -- commit
            stalls = 0
        else:
            # overshoot (crossed zero) or no progress: revert and retry a different candidate
            amorphous_struct.atoms = snapshot
            amorphous_struct.invalidate_graph()
            stalls += 1

    amorphous_struct.sort_atoms()
    final_charge = amorphous_struct.charge()
    if final_charge != 0:
        raise ValueError(
            f"correct_charge could not neutralise the slab: net formal charge {final_charge} "
            f"remains after {iteration} attempt(s). The charge balance is unsatisfiable with "
            f"the available over/under-coordinated sites (e.g. an isolated ion with no "
            f"counter-site to cap).")

    