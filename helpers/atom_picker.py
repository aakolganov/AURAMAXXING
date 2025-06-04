from default_constants import OXIDATION_POS, OXIDATION_NEG

def pick_next_atom(structure, target_ratio: dict) -> str:
    """
    Choosing the atom to add next.:
      1) Calculating the current net charge of the structure.
      2) If positive, add O.
         If negative -> pick Al or Si based on the desired ratio
      3) If ≈ 0, than add the charge based on target ratio
    """
    # 1) charge_neutrality
    q_pos = sum(structure.count_atoms(sym) * ox for sym, ox in OXIDATION_POS.items())
    q_neg = sum(structure.count_atoms(sym) * ox for sym, ox in OXIDATION_NEG.items())
    net_charge = q_pos + q_neg

    # if not compensated -> add O
    if net_charge > 0:
        return "O"

    #) else -> pick Si or Al
    ratios = {}
    for sym, target in target_ratio.items():
        current = structure.count_atoms(sym)
        ratios[sym] = current / target

    # choose the cation atom based on the ratio
    min_val = min(ratios.values())
    candidates = [sym for sym, val in ratios.items() if abs(val - min_val) < 1e-6]
    # if equal - rng
    return structure.rng.choice(candidates)
