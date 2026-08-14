import os
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


def _place_neg_cap(amorphous_struct, anchor_idx: int, bond_lengths, num_samples: int,
                   place_fn) -> None:
    """Place an -OH cap (net -1) on ``anchor_idx``: the O (tagged NEG_CAP) then its H (OH_H),
    each via ``place_fn`` -- the collision-aware ``_try_then_force_place`` in basic saturation,
    or the bystander-aware ``place_atom_terminal`` in charge-correction. The two call sites
    shared this exact O-then-H sequence, so factoring it keeps the cap-role tagging and the RNG
    draw order identical."""
    bl = _cap_bond_length(amorphous_struct, anchor_idx, "O", bond_lengths)
    place_fn(amorphous_struct, "O", anchor_idx, num_samples=num_samples, bond_length=bl)
    o_idx = len(amorphous_struct) - 1
    _set_cap_role(amorphous_struct, o_idx, NEG_CAP)
    bl = _cap_bond_length(amorphous_struct, o_idx, "H", bond_lengths)
    place_fn(amorphous_struct, "H", o_idx, num_samples=num_samples, bond_length=bl)
    _set_cap_role(amorphous_struct, len(amorphous_struct) - 1, OH_H)


def _place_pos_cap(amorphous_struct, anchor_idx: int, bond_lengths, num_samples: int,
                   place_fn) -> None:
    """Place a standalone H cap (net +1, tagged POS_CAP) on ``anchor_idx`` via ``place_fn``."""
    bl = _cap_bond_length(amorphous_struct, anchor_idx, "H", bond_lengths)
    place_fn(amorphous_struct, "H", anchor_idx, num_samples=num_samples, bond_length=bl)
    _set_cap_role(amorphous_struct, len(amorphous_struct) - 1, POS_CAP)


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
    # sorted(): set iteration order depends on PYTHONHASHSEED, and this dict's order decides the
    # element capping order downstream — unsorted, the same seed gives different slabs in
    # different interpreter processes (e.g. spawn workers).
    cn_dict = {at: [] for at in sorted(set(symbols))}
    # under-coordinated: CN below the (per-element) minimum; over-coordinated: CN above
    # the per-atom maximum (so an Al allowed CN 6 isn't flagged over until CN > 6).
    limits = amorphous_struct.min_cn_array() if do_under else amorphous_struct.max_cn_array()
    flagged = (all_cn < limits) if do_under else (all_cn > limits)
    for sym in sorted(set(symbols)):
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
        dump_dir=None,
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

    # Optional per-cap trajectory dump for animating the saturation stage: one xyz frame
    # before capping, then one after each -OH/H cap. Uses atoms.write directly (no sort) so
    # it never perturbs the attach indices collected above.
    _traj = None
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        _traj = os.path.join(str(dump_dir), "saturation.xyz")
        if os.path.exists(_traj):
            os.remove(_traj)
        amorphous_struct.atoms.write(_traj, format="xyz", append=True)

    for sym, idx_list in undr_cn.items():
        is_cation = amorphous_struct.oxidation.get(sym, 0) > 0
        for attach_idx in idx_list:
            if is_cation:
                # negative fragment -OH: O on the cation (tagged NEG_CAP), then H on that O.
                _place_neg_cap(amorphous_struct, attach_idx, bond_lengths, num_samples,
                               _try_then_force_place)
            else:
                # positive fragment H on the anion (tagged POS_CAP).
                _place_pos_cap(amorphous_struct, attach_idx, bond_lengths, num_samples,
                               _try_then_force_place)
            if _traj is not None:
                amorphous_struct.atoms.write(_traj, format="xyz", append=True)


def _break_and_cap_step(amorphous_struct, current_charge, bond_lengths, num_samples, move_alpha):
    """One bond-break + re-cap attempt -- ``correct_charge``'s preferred charge step, which
    repurposes an over/under/variable-CN site so the cap improves coordination rather than over-
    coordinating. The attempt is taken on a snapshot: returns the new net charge if it strictly
    reduced ``abs(charge)`` (commit), otherwise reverts and returns ``None`` so the caller falls
    back to a direct cap. Not snapshotting the rng means the next attempt re-draws a different
    candidate, so the loop can never get stuck overshooting on the same pick."""
    if current_charge > 0:
        # too positive: break a bond at an over-coordinated ANION (oxidation < 0) and add the
        # negative -OH cap to the freed cation neighbour (each cap is net -1).
        over_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=False)
        indices = [i for k, v in over_cn.items()
                   if amorphous_struct.oxidation.get(k, 0) < 0 for i in v]
    else:
        # too negative: break a bond at an under-coordinated CATION (oxidation > 0) and add the
        # positive H cap to the freed anion neighbour (each cap is net +1).
        undr_cn = collect_over_or_under_cn_atoms(amorphous_struct, do_under=True)
        indices = [i for k, v in undr_cn.items()
                   if amorphous_struct.oxidation.get(k, 0) > 0 for i in v]
    if not indices:
        indices = find_tetrogonal_sites(amorphous_struct)
    move = select_idx_for_move(amorphous_struct, indices) if indices else None
    if move is None:
        return None
    chosen_idx_pos, idx_furthest = move

    # Pick the cap anchor by the fragment's required oxidation sign, NOT by which move-pair member
    # select_idx_for_move happened to order first: the -OH cap (current_charge > 0, net -1) must
    # anchor on a CATION and the H cap (net +1) on an ANION. Capping the wrong-sign atom still
    # balances the net charge but builds a peroxide (O-O) or hydride (M-H) motif -- and because the
    # too-positive branch feeds over-coordinated anions (high CN), select_idx_for_move usually
    # returns the anion as idx_furthest, so the old unconditional cap on idx_furthest put -OH on
    # the anion. The moved atom and its pivot are opposite classes in the normal cation<->anion
    # case, so one of them carries the wanted sign; if neither does (a homo-bonded defect), skip and
    # let the caller's direct cap make progress instead of forcing a wrong-sign cap.
    want_cation = current_charge > 0

    def _wanted_sign(i: int) -> bool:
        return (amorphous_struct.oxidation.get(amorphous_struct.atoms[i].symbol, 0) > 0) == want_cation

    anchor = (idx_furthest if _wanted_sign(idx_furthest)
              else chosen_idx_pos if _wanted_sign(chosen_idx_pos) else None)
    if anchor is None:
        return None

    snapshot = amorphous_struct.atoms.copy()
    cn_before = amorphous_struct.get_cn()
    n_before = len(amorphous_struct)
    move_atom(
        amorphous_struct,
        idx_move=chosen_idx_pos,
        move_away_from=idx_furthest,
        dist_move=amorphous_struct.d_min_max[amorphous_struct.atoms[chosen_idx_pos].symbol][amorphous_struct.atoms[idx_furthest].symbol][1]+0.2,
        alpha=move_alpha,
        )

    # Cap the correctly-signed anchor with a bystander-aware placement: the cap bonds only its
    # intended attach atom where possible, so a forced placement can't over-coordinate unrelated
    # atoms (M2).
    if current_charge > 0:
        _place_neg_cap(amorphous_struct, anchor, bond_lengths, num_samples,
                       place_atom_terminal)
    else:
        _place_pos_cap(amorphous_struct, anchor, bond_lengths, num_samples,
                       place_atom_terminal)
    _prune_orphans_from_move(amorphous_struct, cn_before, n_before)

    new_charge = amorphous_struct.charge()
    if abs(new_charge) < abs(current_charge):
        return new_charge
    amorphous_struct.atoms = snapshot
    amorphous_struct.invalidate_graph()
    return None


def _add_charge_cap(amorphous_struct, want_negative: bool, bond_lengths, num_samples: int) -> bool:
    """Charge-correction fallback: cap an existing atom directly to shift the net charge one unit
    toward zero, used when the bond-break strategy has no over/under/variable-CN site to work with
    (e.g. a fully-coordinated fixed-CN oxide). For ``want_negative`` (slab too positive) add an
    -OH group (net -1) to a cation; otherwise add an H (net +1) to an anion. The lowest-coordinated
    eligible atom is chosen, so an under-coordinated site is preferred (the cap then also improves
    coordination) and a fully-coordinated one is only mildly over-coordinated as a last resort. A
    positive net charge guarantees a network cation exists and a negative one an anion, so this
    always makes progress; returns ``False`` only if no network-capable atom of the needed sign
    exists at all (then the slab genuinely has a single charge sign)."""
    symbols = amorphous_struct.symbols
    # Monovalent cap atoms (e.g. the H of an -OH/H cap, max CN 1) carry an oxidation sign but must
    # never anchor a cap: they'd win the lowest-CN tie-break below and the new cap-O would bond an
    # existing hydrogen, leaving a divalent H bridging two oxygens. Only network-capable atoms
    # (per-atom max CN > 1) are eligible.
    max_cn = amorphous_struct.max_cn_array()
    if want_negative:
        cand = [i for i in range(len(symbols))
                if amorphous_struct.oxidation.get(symbols[i], 0) > 0 and max_cn[i] > 1]
    else:
        cand = [i for i in range(len(symbols))
                if amorphous_struct.oxidation.get(symbols[i], 0) < 0 and max_cn[i] > 1]
    if not cand:
        return False
    cn = amorphous_struct.get_cn()
    anchor = int(min(cand, key=lambda i: cn[i]))
    if want_negative:
        _place_neg_cap(amorphous_struct, anchor, bond_lengths, num_samples, place_atom_terminal)
    else:
        _place_pos_cap(amorphous_struct, anchor, bond_lengths, num_samples, place_atom_terminal)
    return True


def correct_charge(
        amorphous_struct: AmorphousStruc,
        bond_lengths=None,
        max_iterations: int = 1000,
        num_samples: int = 250,
        move_alpha: float = 0.5,
        dump_dir=None,
    ):
    """Drive the slab to net formal charge zero by adding H / -OH caps. Two strategies per step:

    1. *Preferred* -- break a bond at an over/under/variable-CN site and re-cap the freed neighbour
       (``_break_and_cap_step``); this repurposes a coordination defect so the cap improves
       coordination rather than over-coordinating. Taken on a snapshot and only committed if it
       strictly reduces ``abs(charge)``, so an orphan-prune that jumps the charge can never make
       the loop overshoot zero.
    2. *Fallback* -- when no such site exists, or the break step can't make progress, cap an
       existing cation (add -OH, slab too positive) or anion (add H, slab too negative) directly
       (``_add_charge_cap``). A positive net charge always has a cation to cap and a negative one
       an anion, so this guarantees one unit of progress toward zero every iteration. The loop
       therefore always reaches 0 -- the cost is mildly over-coordinating a few atoms when the
       structure is already fully coordinated (e.g. a fixed-CN borate / phosphate that the bond-
       break strategy alone could not neutralise). The post-loop check stays as a safety net.
    """
    # Optional per-step trajectory dump for animating the charge-correction stage.
    _traj = None
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        _traj = os.path.join(str(dump_dir), "charge.xyz")
        if os.path.exists(_traj):
            os.remove(_traj)
        amorphous_struct.atoms.write(_traj, format="xyz", append=True)

    current_charge = amorphous_struct.charge()
    iteration = 0
    while current_charge != 0 and iteration < max_iterations:
        iteration += 1
        committed = _break_and_cap_step(amorphous_struct, current_charge,
                                        bond_lengths, num_samples, move_alpha)
        if committed is not None:
            current_charge = committed
            if _traj is not None:
                amorphous_struct.atoms.write(_traj, format="xyz", append=True)
            continue
        # No usable bond-break site (or it could not make progress): fall back to a direct cap,
        # which always shifts the charge one unit toward zero while a cation/anion exists.
        if not _add_charge_cap(amorphous_struct, current_charge > 0, bond_lengths, num_samples):
            break
        current_charge = amorphous_struct.charge()
        if _traj is not None:
            amorphous_struct.atoms.write(_traj, format="xyz", append=True)

    amorphous_struct.sort_atoms()
    final_charge = amorphous_struct.charge()
    if final_charge != 0:
        raise ValueError(
            f"correct_charge could not neutralise the slab: net formal charge {final_charge} "
            f"remains after {iteration} iteration(s) -- no atom of the needed charge sign was "
            f"available to cap (the structure has only one charge sign).")


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


# --- post-growth network completion ("stitching") ---------------------------------------

def _near_growth_face(amorphous_struct, positions, margin: float) -> np.ndarray:
    """True for atoms within ``margin`` of the top or bottom growth boundary at their own
    (x, y) grid cell. Both boundaries are genuine surfaces: their dangling anions are the
    surface chemistry that the saturation stage caps, so stitching must leave them alone.
    Without installed limits nothing is classified as a face."""
    lim = amorphous_struct.limits
    mask = np.zeros(len(positions), dtype=bool)
    if lim is None:
        return mask
    ix = np.clip((positions[:, 0] / lim.dx).astype(int), 0, lim.nx - 1)
    iy = np.clip((positions[:, 1] / lim.dy).astype(int), 0, lim.ny - 1)
    z = positions[:, 2]
    if lim.upper_lim is not None:
        mask |= z > (lim.upper_lim[ix, iy] - margin)
    if lim.lower_lim is not None:
        mask |= z < (lim.lower_lim[ix, iy] + margin)
    return mask


def stitch_network(
        amorphous_struct: AmorphousStruc,
        r_max: float = 4.0,
        surface_margin: float = 2.5,
    ) -> int:
    """Geometric network completion: run after growth, before the first relax.

    The as-grown network leaves a large population of singly-bonded anions that are NOT
    at a surface (interior porosity plus none of them yet capped); handed to the first
    MLIP relax in that state, the optimizer completes the network itself -- partly by
    condensing nearby dangling anions into O-O / M-M homo bonds. This pass does the
    completion geometrically instead: each interior dangling anion (CN below its minimum,
    outside ``surface_margin`` of both growth boundaries) is pulled onto the nearest
    not-already-bonded cation with spare capacity (CN below its per-atom maximum) within
    ``r_max``, using the damped ``move_atom`` displacement at the covalent-radii bond
    length. A move that fails to form the bond or violates a collision floor is reverted
    and that anion is left for the saturation stage.

    Deterministic and rng-free (globally nearest pair first, ties by index), so enabling
    it does not perturb any downstream RNG stream. Returns the number of bonds formed.
    """
    formed = 0
    failed: set = set()
    while True:
        cn = amorphous_struct.get_cn()
        min_cn = amorphous_struct.min_cn_array()
        max_cn = amorphous_struct.max_cn_array()
        syms = np.array(amorphous_struct.symbols)
        ox = np.array([amorphous_struct.oxidation.get(s, 0) for s in syms])
        pos = amorphous_struct.atoms.get_positions()
        face = _near_growth_face(amorphous_struct, pos, surface_margin)

        dang = np.array([i for i in np.where((ox < 0) & (cn < min_cn) & ~face)[0]
                         if i not in failed], dtype=int)
        cap = np.where((ox > 0) & (cn < max_cn))[0]
        if len(dang) == 0 or len(cap) == 0:
            break

        graph = amorphous_struct.get_graph()
        homo_before = sum(1 for u, v in graph.edges if syms[u] == syms[v])
        cell = amorphous_struct.atoms.cell.lengths()
        d = pos[dang][:, None, :] - pos[cap][None, :, :]
        d -= np.round(d / cell) * cell
        dist = np.sqrt((d ** 2).sum(-1))
        for a, i in enumerate(dang):                      # mask already-bonded pairs
            for b, j in enumerate(cap):
                if graph.has_edge(int(i), int(j)):
                    dist[a, b] = np.inf
        best = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[best] > r_max:
            break
        o_idx, c_idx = int(dang[best[0]]), int(cap[best[1]])

        snapshot = amorphous_struct.atoms.copy()
        blen = derive_bond_length(syms[o_idx], syms[c_idx],
                                  bond_factor=getattr(amorphous_struct, "bond_factor", None))
        # A distant stitch needs the anion's old neighbourhood to follow: move_atom's spring
        # relaxation restores the old bond by dragging the anion back, so a single call can
        # leave the new bond just outside the cutoff. Repeat the damped move (each round pulls
        # the old neighbours along) until the bond registers, up to 3 rounds.
        for _ in range(3):
            move_atom(amorphous_struct, idx_move=o_idx, move_away_from=c_idx, dist_move=blen)
            if amorphous_struct.get_graph(force_rebuild=True).has_edge(o_idx, c_idx):
                break

        # Accept only if the bond really formed, neither end became over-coordinated, no new
        # homo-element bond appeared anywhere (the defect class this pass exists to prevent),
        # and the moved anion violates no collision floor against anything it is not bonded to.
        new_graph = amorphous_struct.get_graph()
        new_cn = amorphous_struct.get_cn()
        ok = (new_graph.has_edge(o_idx, c_idx)
              and new_cn[o_idx] <= max_cn[o_idx] and new_cn[c_idx] <= max_cn[c_idx]
              and sum(1 for u, v in new_graph.edges if syms[u] == syms[v]) <= homo_before)
        if ok:
            new_pos = amorphous_struct.atoms.get_positions()
            vec = new_pos[o_idx] - new_pos
            vec -= np.round(vec / cell) * cell
            dd = np.sqrt((vec ** 2).sum(-1))
            for j in range(len(dd)):
                if j == o_idx or new_graph.has_edge(o_idx, j):
                    continue
                floor = amorphous_struct.d_min_max[syms[o_idx]][syms[j]][0]
                if dd[j] < floor:
                    ok = False
                    break
        if ok:
            formed += 1
        else:
            amorphous_struct.atoms = snapshot
            amorphous_struct.invalidate_graph()
            failed.add(o_idx)
    return formed
