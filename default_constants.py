from typing import Dict, List, Callable
from helpers.random_sample import RandomSample
from scipy.stats import burr12, uniform


# oxidation states
OXIDATION_POS = {"Si": +4, "Al": +3, "H": +1}
OXIDATION_NEG = {"O": -2}


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
        "Al": uniform(loc=1.8, scale=0.4)
    }),
    "Al": RandomSample({
        "Si": uniform(loc=2.2, scale=0.32),
        "O": uniform(loc=1.8, scale=0.4),
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


# Overcoordination policy: per element, allow a random fraction of atoms to use an
# elevated max coordination number. {} means the feature is off (every atom uses the
# per-element max_cn). Each eligible atom is independently tagged at creation with
# probability "fraction" (Bernoulli), so the realised fraction is approximate.
default_overcoord_policy: Dict[str, Dict] = {}

# Example policy used by the Siral generation scripts: ~20% of Al atoms may reach CN 6
# (octahedral), the rest stay capped at the standard CN 4 (tetrahedral).
SIRAL_OVERCOORD: Dict[str, Dict] = {"Al": {"max_cn": 6, "fraction": 0.2}}
