from default_constants import OXIDATION_POS, OXIDATION_NEG
import numpy as np
from base import AmorphousStruc
from typing import Optional


def choose_atom_idx_to_attach_to(amorphous_struct: AmorphousStruc, atom_type: str, weight_z: bool = True, added_weights: Optional[callable]=None, exclude_indices: Optional[list[int]] = None) -> int:
    symbols = np.array(amorphous_struct.symbols)
    all_cn = amorphous_struct.get_cn()

    # Determine which atom types are valid to attach to
    allowed_attachment_atoms = OXIDATION_NEG.keys() if atom_type in OXIDATION_POS else OXIDATION_POS.keys()

    # Build a mask of saturated atoms
    is_saturated = np.zeros(len(symbols), dtype=bool)
    for sym, max_cn in amorphous_struct.max_cn.items():
        is_saturated[symbols == sym] = all_cn[symbols == sym] >= max_cn

    # Candidates are atoms of allowed types that are not saturated
    cand = np.where(np.isin(symbols, list(allowed_attachment_atoms)) & ~is_saturated)[0]

    if exclude_indices is not None:
        cand = np.setdiff1d(cand, exclude_indices)

    # if none left, pick uniformly at random from all atoms
    if cand.size == 0:
        if exclude_indices is not None:
            all_indices = np.arange(len(amorphous_struct))
            fallback_cand = np.setdiff1d(all_indices, exclude_indices)
            if fallback_cand.size > 0:
                return int(amorphous_struct.rng.choice(fallback_cand))
        return int(amorphous_struct.rng.integers(len(amorphous_struct)))
    
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

    symbols = np.array(amorphous_struct.symbols)
    current_frac = {}
    for symbol in target_ratio.keys():
        current_frac[symbol] = np.count_nonzero(symbols == symbol) / num_atoms

    deficits = {
        symbol: target_frac[symbol] - current_frac.get(symbol, 0.0)
        for symbol in target_ratio.keys()
    }
    return max(deficits, key=deficits.get)