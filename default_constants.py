from typing import Dict, List, Callable
from helpers.random_sample import RandomSample
from scipy.stats import burr12, uniform


# oxidation states
OXIDATION_POS = {"Si": +4, "Al": +3, "H": +1}
OXIDATION_NEG = {"O": -2}

# Merged signed oxidation table (sign defines cation vs anion). Default for
# AmorphousStruc.charge() and the growth cation/anion attachment rule; a run with other
# elements overrides this via the config/element_data derivation.
default_oxidation = {**OXIDATION_NEG, **OXIDATION_POS}


# object to randomly select the distances
sample_dist: Dict[str, RandomSample[str, Callable]] = {
    "Si": RandomSample({
        "Si": burr12(c=20.50918422948114, d=3.282331385061921, loc=1.8153399428512698, scale=1.3978541397862818),
        "O": uniform(loc=1.6, scale=0.32),
        "Al": uniform(loc=2.2, scale=0.32)
    }),

    "O": RandomSample({
        "Si": uniform(loc=1.6, scale=0.32),
        # O-O is never sampled during growth (cation-anion attachment only); use a
        # plain positive window matching d_min_max instead of the original fitted
        # burr12 whose loc=-0.226 admitted negative distances.
        "O": uniform(loc=2.05, scale=0.35),
        # [1.7, 2.1): upper bound == the (O,Al) bonding cutoff (default_max_cut_offs)
        # so a sampled Al-O placement is always inside the cutoff and is counted as a
        # graph edge. The previous [1.8, 2.2] over-shot the 2.1 cutoff ~25% of the time,
        # committing "bonded" atoms the coordination graph then scored as non-bonded.
        "Al": uniform(loc=1.7, scale=0.4)
    }),
    "Al": RandomSample({
        "Si": uniform(loc=2.2, scale=0.32),
        # [1.7, 2.1): see the O->Al note above -- upper bound == the (O,Al) cutoff so the
        # placement is always a graph edge (was [1.8, 2.2], over-shooting the 2.1 cutoff).
        "O": uniform(loc=1.7, scale=0.4),
        "Al": burr12(c=20.50918422948114, d=3.282331385061921, loc=1.8153399428512698, scale=1.3978541397862818)
    })
}

# range of distances for seeded growth
d_min_max: Dict[str, Dict[str, List]] = {
    "Si": {"Si": [2.6, 3.0], "O": [1.585, 1.92], "Al": [2.6, 3.0], "H": [1.65, 1.85]},
    "O": {"Si": [1.585, 1.92], "O": [2.05, 2.4], "Al": [1.7, 2.2], "H": [0.90, 1.2]},
    "Al": {"Si": [2.6, 3.0], "O": [1.7, 2.2], "Al": [2.6, 3.0], "H": [1.55, 2.0]},
    "H": {"Si": [1.65, 1.85], "O": [0.90, 1.2], "Al": [1.55, 2.0], "H": [0.7, 0.8]},
}


# maximum allowed coordination number for each atom type
default_max_cn = {"Si": 4, "O": 2, "Al": 4, "H": 1}  # we consider Al as tetragonal
default_min_cn = {"Si": 4, "O": 2, "Al": 3, "H": 1}  # we consider Al as tetragonal


default_max_cut_offs = {
    ('Si','Si'): 2.3, ('Si','O'): 2.0, ('Si','Al'): 2.3, ('Si','H'): 1.5,
     ('O','O'): 1.8, ('O','Al'): 2.1, ('O','H'): 1.1,
     ('Al','Al'): 2.3, ('Al','H'): 1.7,('H','H'): 0.7}

pair_cutoffs = {}
for (a,b), val in default_max_cut_offs.items():
    pair_cutoffs[(a,b)] = val
    pair_cutoffs[(b,a)] = val

ev_to_kcal = 23.060541945

default_masses = {
    "Si": 28.085,
    "O": 15.9999,
    "Al": 27.0,
    "H": 1.008
}
# Only the Legacy LAMMPS interface (Legacy/LAMMPS_Interface.py) reads these. The active
# interfaces.LAMMPS_Interface uses bks_charges, and AmorphousStruc.charge() uses
# OXIDATION_POS/NEG -- so this table is not live config; kept in sync with bks_charges.
default_charges = {
    "Si": 2.4,
    "O": -1.2,
    "Al": 1.8,  # matches bks_charges; keeps Al2O3 charge-neutral (2*1.8 - 3*1.2 = 0)
    "H": 1.2
}

bks_params = {
    # buck/coul/long: (A_ij, eV; b_ij, A^-1, c_ij, eV*A^6)
    ("Si", "Si"): (834.40, 1/0.29, 0.0),
    ("O",  "O" ): (1388.7730, 2.76000,175.0000),
    ("Si", "O" ): (45296.72, 1/0.161,46.1395),
    ("Si", "Al"): (646.67, 1/0.120, 0.0),
    ("O",  "Al"): (28287.00, 1/0.172, 34.7600),
    ("Al", "Al"): (351.94, 1/0.360, 0.0),
}
# lj/cut: (epsilon, sigma, sigma)
bks_lj = {
    ("Si", "Si"): ( 0.0, 0.0, 0.0),
    ("O",  "O" ): (2.6, 1.6, 1.6),
    ("Si", "O" ): (2.0, 1.2, 1.2),
    ("Si", "Al"): (0.0, 0.0, 0.0),
    ("O",  "Al"): (2.3, 1.4, 1.4),
    ("Al", "Al"): ( 0.0, 0.0, 0.0),
}
# charges
bks_charges = {
    "Si":  2.4,
    "O" : -1.2,
    "Al":  1.8,
}

# bond lenghst for satuarion routine
O_H_BONDLENGTH = 0.98
El_O_BONDLENGTH = 1.63 #bond lenght for Si/Al - O bond

# If the over-coordinated atoms formally gets assigned a positive charge then it is true
OVER_POS = {"Si": False, "O": True, "H": False, "Al": False}


# Coordination-number distribution (cn_distr): per element, the expected coordination
# number(s) and their fractions. {} means "use the curated per-element default CN" for every
# element. Each atom is independently assigned a CN at creation by sampling the distribution
# (one draw), so the realised fractions are approximate. Accepted per-element input forms:
#   Si: 4                       # a single expected CN -> all Si are CN 4
#   Al: {4: 0.8, 6: 0.2}        # 80% CN 4, 20% CN 6
#   Si: [{cn: 4, fraction: 0.9}, {cn: 3, fraction: 0.1, oxidation: 3}]   # CN + oxidation
# A CN sets the atom's max coordination; an optional per-variant oxidation overrides the
# element's default oxidation state for that same fraction. Any unspecified remaining fraction
# (sum < 1) falls back to the curated element default.
default_cn_distr: Dict[str, object] = {}

# Example used by the Siral generation scripts: ~20% of Al grown as CN 6 (octahedral), the
# rest at the standard CN 4 (tetrahedral).
SIRAL_CN_DISTR: Dict[str, object] = {"Al": {6: 0.2}}
