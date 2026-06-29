import numpy as np
from base import AmorphousStruc
from typing import Optional


def choose_atom_idx_to_attach_to(amorphous_struct: AmorphousStruc, atom_type: str, weight_z: bool = True, added_weights: Optional[callable]=None, exclude_indices: Optional[list[int]] = None, all_cn: Optional[np.ndarray] = None) -> int:
    symbols = np.array(amorphous_struct.symbols)
    # Coordination numbers don't change across retries on the same structure, so a
    # caller in a retry loop can compute them once and pass them in to avoid the
    # repeated all-atom graph walk.
    if all_cn is None:
        all_cn = amorphous_struct.get_cn()

    # Determine which atom types are valid to attach to: the opposite oxidation-state sign
    # (cations attach to anions and vice versa), read from the structure's oxidation table.
    ox = amorphous_struct.oxidation
    if atom_type not in ox:
        raise ValueError(f"choose_atom_idx_to_attach_to: no oxidation state for {atom_type!r}")
    adding_is_cation = ox[atom_type] > 0
    allowed_attachment_atoms = [el for el, q in ox.items() if (q > 0) != adding_is_cation]

    # Build a mask of saturated atoms. max_cn_array() is per-atom: it honours any
    # cn_distr overrides (e.g. an Al grown at CN 6) and falls back to the per-element
    # max_cn otherwise.
    is_saturated = all_cn >= amorphous_struct.max_cn_array()

    # Candidates are atoms of allowed types that are not saturated
    cand = np.where(np.isin(symbols, list(allowed_attachment_atoms)) & ~is_saturated)[0]

    if exclude_indices is not None:
        cand = np.setdiff1d(cand, exclude_indices)

    if cand.size == 0:
        # No unsaturated opposite-class anchor. Fall back to ANY opposite-class atom (even a
        # saturated one) so we never bond to a same-class anchor -- whose bond-length sampler
        # / cutoff would score the placement as non-bonded (committing a floating atom). If no
        # opposite-class atom exists at all, signal failure (-1) so the growth loop relieves
        # the jam by melt-quenching instead of committing a spurious bond.
        pool = np.where(np.isin(symbols, list(allowed_attachment_atoms)))[0]
        if exclude_indices is not None:
            pool = np.setdiff1d(pool, exclude_indices)
        if pool.size == 0:
            return -1
        return int(amorphous_struct.rng.choice(pool))
    
    # 4) group by coordination number
    cand_cns = all_cn[cand]
    unique_cns = np.unique(cand_cns)
    # prefer highest CN
    weights = 2 ** (unique_cns + 1)
    probs = weights/weights.sum()
    pick_cn = amorphous_struct.rng.choice(unique_cns, p=probs)

    # subset of indexes to attach
    sub = cand[cand_cns == pick_cn]

    # base weights = 1
    w = np.ones(len(sub), dtype=float)

    # if we want to weight also by z-coordinate:
    if weight_z:
        zpos = amorphous_struct.atoms.get_positions()[sub, 2]
        w = np.exp(-((zpos - zpos.mean()) ** 2))
        w = np.ones_like(zpos) if np.allclose(w, 0) else w / w.sum()
    
    if added_weights is not None:
        w = added_weights(amorphous_struct, w, sub)

    # Renormalize & choose
    total = w.sum()
    if total <= 0:
        # fallback to uniform
        w = np.ones_like(w)
    w /= w.sum()
    output = int(amorphous_struct.rng.choice(sub, p=w))
    return output


def pick_next_atom_type(amorphous_struct: AmorphousStruc, target_ratio: dict[str, float]) -> str:
    num_atoms = len(amorphous_struct)
    if num_atoms == 0:
        return amorphous_struct.rng.choice(list(target_ratio.keys()))

    total = sum(target_ratio.values())
    target_frac = {}
    for key, val in target_ratio.items():
        target_frac[key] = val / total

    # Fractions are taken over the target elements only, so pre-existing atoms of other
    # elements (e.g. a loaded substrate of a foreign element) don't skew the balance.
    symbols = np.array(amorphous_struct.symbols)
    target_counts = {s: int(np.count_nonzero(symbols == s)) for s in target_ratio}
    target_total = sum(target_counts.values()) or 1
    current_frac = {s: target_counts[s] / target_total for s in target_ratio}

    deficits = {
        symbol: target_frac[symbol] - current_frac.get(symbol, 0.0)
        for symbol in target_ratio.keys()
    }
    return max(deficits, key=deficits.get)