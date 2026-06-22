"""
Saturate a grown (but bare / under-coordinated) Siral-10 slab with -OH / =O groups
and relax it with a MACE potential.

Pipeline (current API):
  1. load the structure from a POSCAR,
  2. add H/OH to under-coordinated atoms (saturate_under_coordinated),
  3. relax with MACE (finalize_structure),
  4. neutralise the net charge (correct_charge) and relax again.

Run from this directory so the POSCAR and the .model file are found.
"""
from base.initialize import initialize_structure_file
from interfaces.MACE_interface import MACEInterface
from saturation.new_sat import saturate_under_coordinated, correct_charge
from growth.new_growth import finalize_structure


def main():
    struct = initialize_structure_file("POSCAR_Siral_10", ase_read_kwargs={})

    # NOTE on MPS: MACEInterface loads the model in float32 (MACE checkpoints are
    # often float64, which MPS cannot load directly, so it casts on CPU first).
    # mace-torch 0.3.16's forward still computes some terms in float64, which MPS
    # does not support, so the relaxation may fail on MPS with that version --
    # use device="cpu"/"cuda" (or a newer mace) if you hit that.
    calc = MACEInterface(
        mace_model_path="2024-01-07-mace-128-L2_epoch-199.model",  # path to the MACE model
        device="mps",          # "cuda" / "cpu" / "mps" (Apple Silicon)
        dtype="float32",       # load/run the model in float32 (also required for MPS)
        dump_path="dump_mace",
    )

    # 1) add H / OH to under-coordinated atoms, then relax
    saturate_under_coordinated(struct)
    finalize_structure(struct, calculator=calc)

    # 2) make the surface charge-neutral, then relax again
    correct_charge(struct)
    finalize_structure(struct, calculator=calc)

    struct.atoms.write("dump_mace/saturated_siral_10.vasp", format="vasp")


if __name__ == "__main__":
    main()
