from growth.new_growth import grow_structure, finalize_structure
from base.initialize import initialize_structure_blank
from base.limits import make_limit_flat, make_limits_fourier, fix_limits
from interfaces.MACE_interface import MACEInterface
from interfaces.LAMMPS_Interface import LMPInterface
from saturation.new_sat import saturate_under_coordinated, correct_charge
from pathlib import Path
from rules import PeriodicStructureModifier, AvoidMotifSwapRule

def main():
    # calc = MACEInterface(
    #     mace_model_path="test_model/MACE-matpes-r2scan-omat-ft.model",
    #     device="cuda",
    #     use_dispersion=True,
    #     dispersion_xc="r2scan",
    # )
    calc = LMPInterface(dump_path="dump_lmp")
    struct = initialize_structure_blank(cell=[20, 20, 40])
    struct.set_seed(42)

    make_limit_flat(struct, z_val=15, is_for="bottom")
    make_limits_fourier(struct, z_av=20, alpha=0.1, is_for="top")
    fix_limits(struct.limits, hard_limit="bottom")

    grow_structure(
        amorphous_struct=struct,
        target_number_atoms=3*96,
        target_ratios={"Si":80, "Al": 16, "O":184},
        calculator=calc,
        output_dir=Path("growth_test")
    )
    finalize_structure(
        amorphous_struct=struct,
        calculator=calc,
    )

    calc = MACEInterface(
        dump_path="dump_mace",
        mace_model_path="test_model/MACE-matpes-r2scan-omat-ft.model",
        device="cuda",
        use_dispersion=True,
        dispersion_xc="r2scan",
    )
    
    saturate_under_coordinated(
        amorphous_struct=struct,
    )
    struct.atoms.write("dump_mace/final.vasp", format="vasp")

    finalize_structure(
        amorphous_struct=struct,
        calculator=calc,
    )
    correct_charge(
        amorphous_struct=struct
    )
    finalize_structure(
        amorphous_struct=struct,
        calculator=calc,
    )

    modifier = PeriodicStructureModifier(struct)
    modifier.add_rule(AvoidMotifSwapRule(edge_element="Al", center_element="O", swap_candidate="Si"))
    is_valid = modifier.optimize(max_iterations=500)
    finalize_structure(
        amorphous_struct=struct,
        calculator=calc,
    )
    
if __name__ == "__main__":
    main()