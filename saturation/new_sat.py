from typing import Optional

from base.amorphous_structure import AmorphousStruc
import numpy as np
from helpers.files_io import highlight_coordination
from ase.geometry import find_mic
from helpers.atom_placing import place_atom_sphere, place_atom_force, place_atom_terminal
from base.element_data import derive_bond_length

# Cap-provenance tags stored per-atom in atoms.arrays["cap_role"] (0 = not a cap). They mark
# which atoms were added as saturation caps so a later relabel pass can swap cap identities
# unambiguously: NEG_CAP is the oxygen of an -OH group placed on a cation (an OH group is a
# net -1 fragment), OH_H is that group's hydrogen, and POS_CAP is a standalone H placed on an
# anion (a net +1 fragment). Tagging the -OH hydrogen explicitly (rather than finding it via
# the bonding graph) lets the relabel balance an OH->F swap by *count* -- retype every NEG_CAP
# O and drop every OH_H, equal in number by construction -- so a stretched O-H bond can't break
# charge neutrality.
CAP_NONE = 0
POS_CAP = 1
NEG_CAP = 2
OH_H = 3


def _set_cap_role(amorphous_struct, idx: int, role: int) -> None:
    """Tag atom ``idx`` with a cap-provenance ``role`` (CAP_NONE/POS_CAP/NEG_CAP), creating
    the per-atom ``cap_role`` array on first use. ASE zero-pads it on later appends, so an
    untagged atom always reads back as CAP_NONE."""
    if amorphous_struct.atoms.arrays.get("cap_role") is None:
        amorphous_struct.atoms.set_array("cap_role",
                                         np.zeros(len(amorphous_struct.atoms), dtype=int))
    amorphous_struct.atoms.arrays["cap_role"][idx] = role


def _cap_bond_length(amorphous_struct, anchor_idx: int, cap_sym: str, override) -> float:
    """Bond length for placing ``cap_sym`` onto the atom at ``anchor_idx``: derived from
    covalent radii for the (anchor, cap) element pair, unless ``override`` (a {cap: length}
    dict) supplies one for ``cap_sym``. Per-pair derivation works for any element pair,
    including X-H / X-F pairs the run's ``sample_dist`` never carries."""
    if override is not None and cap_sym in override:
        return override[cap_sym]
    return derive_bond_length(amorphous_struct.atoms[anchor_idx].symbol, cap_sym,
                              bond_factor=getattr(amorphous_struct, "bond_factor", None))


def _try_then_force_place(amorphous_struct, place_atom: str, attach_idx: int, *,
                          num_samples: int, bond_length: float) -> None:
    """Attach ``place_atom`` to ``attach_idx`` at ``bond_length``: try the collision-aware
    spherical placement first, and fall back to the always-succeeds least-overlap placement
    if that is sterically blocked, so a cap is always added."""
    if not place_atom_sphere(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx,
                             num_samples=num_samples, bond_length=bond_length):
        place_atom_force(amorphous_struct, atom_type=place_atom, idx_anchor=attach_idx,
                         num_samples=num_samples, bond_length=bond_length)


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
    """Basic saturation: cap each under-coordinated cation (positive oxidation) with an -OH
    group and each under-coordinated anion (negative oxidation) with an H, and tag the added
    atoms with their cap role. Does not optimize the structure. Cation vs anion is read from
    the struct's per-element ``oxidation`` (data-driven, not a hardcoded Si/Al/O/H table);
    ``bond_lengths`` optionally overrides the covalent-radii-derived cap bond lengths."""
    amorphous_struct.atoms.wrap()

    # Optional debug dump highlighting coordination defects; off by default so production
    # runs don't write a file into the working directory.
    if highlight_file is not None:
        highlight_coordination(amorphous_struct, highlight_file)
    undr_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=True)

    for sym, idx_list in undr_cn.items():
        is_cation = amorphous_struct.oxidation.get(sym, 0) > 0
        for attach_idx in idx_list:
            if is_cation:
                # negative fragment -OH: O on the cation (tagged NEG_CAP), then H on that O.
                bl = _cap_bond_length(amorphous_struct, attach_idx, "O", bond_lengths)
                _try_then_force_place(amorphous_struct, "O", attach_idx,
                                      num_samples=num_samples, bond_length=bl)
                attach_idx = len(amorphous_struct) - 1   # to account for the 0 index
                _set_cap_role(amorphous_struct, attach_idx, NEG_CAP)
                bl = _cap_bond_length(amorphous_struct, attach_idx, "H", bond_lengths)
                _try_then_force_place(amorphous_struct, "H", attach_idx,
                                      num_samples=num_samples, bond_length=bl)
                _set_cap_role(amorphous_struct, len(amorphous_struct) - 1, OH_H)
            else:
                # positive fragment H on the anion (tagged POS_CAP).
                bl = _cap_bond_length(amorphous_struct, attach_idx, "H", bond_lengths)
                _try_then_force_place(amorphous_struct, "H", attach_idx,
                                      num_samples=num_samples, bond_length=bl)
                _set_cap_role(amorphous_struct, len(amorphous_struct) - 1, POS_CAP)


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
    current_charge = amorphous_struct.charge()
    iteration = 0
    stalls = 0
    max_stalls = max(20, 4 * len(amorphous_struct))
    while current_charge != 0:
        if iteration >= max_iterations or stalls >= max_stalls:
            break
        iteration += 1
        if current_charge > 0:
            # too positive: break a bond at an over-coordinated ANION (oxidation < 0) and add
            # the negative -OH cap to the freed cation neighbour (each cap is net -1).
            over_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=False)
            indices = [i for k, v in over_cn.items()
                       if amorphous_struct.oxidation.get(k, 0) < 0 for i in v]
        else:
            # too negative: break a bond at an under-coordinated CATION (oxidation > 0) and add
            # the positive H cap to the freed anion neighbour (each cap is net +1).
            undr_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=True)
            indices = [i for k, v in undr_cn.items()
                       if amorphous_struct.oxidation.get(k, 0) > 0 for i in v]

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
            dist_move=amorphous_struct.d_min_max[amorphous_struct.atoms[chosen_idx_pos].symbol][amorphous_struct.atoms[idx_furthest].symbol][0]+0.2,
            alpha=move_alpha,
            )

        # Cap with a bystander-aware placement: the cap bonds only its intended attach atom
        # where possible, so a forced placement can't over-coordinate unrelated atoms (M2).
        # (The attach atom itself may still gain a bond beyond its max -- the move above was
        # meant to free a slot but is left as a no-op, so this only guards the bystanders.)
        attach_idx = idx_furthest
        if current_charge > 0:
            bl = _cap_bond_length(amorphous_struct, attach_idx, "O", bond_lengths)
            place_atom_terminal(amorphous_struct, "O", attach_idx,
                                bond_length=bl, num_samples=num_samples)
            attach_idx = len(amorphous_struct) - 1
            _set_cap_role(amorphous_struct, attach_idx, NEG_CAP)
            bl = _cap_bond_length(amorphous_struct, attach_idx, "H", bond_lengths)
            place_atom_terminal(amorphous_struct, "H", attach_idx,
                                bond_length=bl, num_samples=num_samples)
            _set_cap_role(amorphous_struct, len(amorphous_struct) - 1, OH_H)
        else:
            bl = _cap_bond_length(amorphous_struct, attach_idx, "H", bond_lengths)
            place_atom_terminal(amorphous_struct, "H", attach_idx,
                                bond_length=bl, num_samples=num_samples)
            _set_cap_role(amorphous_struct, len(amorphous_struct) - 1, POS_CAP)
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


def _relabel_target(fragment, default_chain):
    """The single replacement element for a fragment, or ``None`` when it is the default chain
    (so no relabel happens). ``fragment`` is an ordered element tuple/list, or ``None``."""
    if fragment is None:
        return None
    frag = tuple(fragment)
    if frag == tuple(default_chain):
        return None
    return frag[0]   # non-default fragments are a single atom (validated at config load)


def _retype_and_reposition(amorphous_struct, idx, anchor_idx, target_sym, pos0, bond_lengths):
    """Retype atom ``idx`` to ``target_sym``. With an ``anchor_idx`` (its surface partner) move
    it to the derived anchor-target bond length along the original anchor->idx direction (MIC);
    with ``anchor_idx is None`` (anchor not in the graph, e.g. a stretched bond) retype in place
    and let the post-relabel relax fix the geometry. ``pos0`` is the position snapshot taken
    before any relabel edit (anchors are network atoms, never retyped, so their rows stay valid).
    Position is cosmetic only -- it never affects the formal charge."""
    if anchor_idx is None:
        amorphous_struct.replace_atom(target_sym, pos0[idx], idx)
        return
    cell = amorphous_struct.atoms.get_cell()
    pbc = amorphous_struct.atoms.get_pbc()
    vec, dist = find_mic(pos0[idx] - pos0[anchor_idx], cell, pbc)
    bl = bond_lengths.get(target_sym) if bond_lengths else None
    if bl is None:
        bl = derive_bond_length(amorphous_struct.atoms[anchor_idx].symbol, target_sym,
                                bond_factor=getattr(amorphous_struct, "bond_factor", None))
    if dist < 1e-6:
        direction = amorphous_struct.rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
    else:
        direction = vec / dist
    amorphous_struct.replace_atom(target_sym, pos0[anchor_idx] + direction * bl, idx)


def relabel_caps(amorphous_struct, negative_fragment=None, positive_fragment=None,
                 bond_lengths=None):
    """Swap the engine's reference caps for the run's 1-valent fragments, preserving the net
    formal charge. The saturation/charge-correction stage always caps with the validated
    reference fragments -OH (on cations, net -1) and H (on anions, net +1) and tags the added
    atoms via ``cap_role`` (NEG_CAP oxygen + its OH_H hydrogen; POS_CAP hydrogen). Here:

    - ``negative_fragment`` (a single anion element, e.g. "F") replaces each -OH group: every
      NEG_CAP oxygen is retyped to the target and every OH_H hydrogen is removed. The two tag
      sets are equal in number by construction (one H per O), so the swap is charge-balanced by
      *count* -- it does not depend on pairing O to H through the bonding graph (a stretched
      O-H bond used to leave an O retyped but its H not removed, breaking neutrality).
    - ``positive_fragment`` (a single cation element, e.g. "Na") replaces each standalone H cap:
      every POS_CAP hydrogen is retyped to the target (1-for-1, +1 -> +1).

    The OH/H defaults (``("O","H")`` / ``("H",)`` or ``None``) mean "leave as is", so the
    default path is a no-op. Raises if the substitution did not preserve the net charge."""
    target_neg = _relabel_target(negative_fragment, ("O", "H"))
    target_pos = _relabel_target(positive_fragment, ("H",))
    if target_neg is None and target_pos is None:
        return

    roles = amorphous_struct.atoms.arrays.get("cap_role")
    if roles is None:
        return
    roles = np.asarray(roles)
    graph = amorphous_struct.get_graph(force_rebuild=True)
    pos0 = amorphous_struct.atoms.get_positions()
    symbols = amorphous_struct.symbols
    charge_before = amorphous_struct.charge()

    # Retypes keep indices stable (anchors are network atoms, never retyped); the OH hydrogens
    # are removed in one batch at the end, so indices recorded here stay valid throughout.
    remove = set()
    if target_neg is not None:
        # Swap each clean -OH group (one terminal cap-O + its hydrogen) for the target, pairing
        # each retyped O with exactly one OH_H to drop so the swap is charge-balanced one-for-one.
        # We pair via the live O-H bond when it survived, else fall back to any unclaimed OH_H by
        # count -- so neither a stretched O-H bond NOR an asymmetric orphan-prune (which can delete
        # a NEG_CAP O or its OH_H independently, diverging the counts) can break neutrality. A
        # cap-O the relax pulled into a bridge (>=1 means !=1 cation neighbour) is NOT a clean
        # terminal cap, so it is left as oxide/hydroxyl rather than retyped to a 1-valent anion
        # bonded to two cations.
        oh_pool = [int(i) for i in np.where(roles == OH_H)[0]]
        for i in np.where(roles == NEG_CAP)[0]:
            i = int(i)
            if len(remove) >= len(oh_pool):
                break                               # no OH_H left to balance another retype
            cats = [n for n in graph.neighbors(i) if symbols[n] != "H"
                    and amorphous_struct.oxidation.get(symbols[n], 0) > 0]
            if len(cats) != 1:
                continue                            # bridging / isolated cap-O -> leave as is
            _retype_and_reposition(amorphous_struct, i, cats[0], target_neg, pos0, bond_lengths)
            own = next((n for n in graph.neighbors(i)
                        if roles[n] == OH_H and n not in remove), None)
            if own is None:
                own = next((h for h in oh_pool if h not in remove), None)
            remove.add(own)
    if target_pos is not None:
        for i in np.where(roles == POS_CAP)[0]:
            i = int(i)
            anchor = next((n for n in graph.neighbors(i)
                           if amorphous_struct.oxidation.get(symbols[n], 0) < 0), None)
            _retype_and_reposition(amorphous_struct, i, anchor, target_pos, pos0, bond_lengths)

    if remove:
        mask = np.zeros(len(amorphous_struct), dtype=bool)
        mask[list(remove)] = True
        amorphous_struct.remove_atom(mask)

    amorphous_struct.invalidate_graph()
    final_charge = amorphous_struct.charge()
    if final_charge != charge_before:
        raise ValueError(
            f"relabel_caps changed the net formal charge from {charge_before} to {final_charge}; "
            f"cap substitution must be charge-preserving (1-valent fragments only).")
