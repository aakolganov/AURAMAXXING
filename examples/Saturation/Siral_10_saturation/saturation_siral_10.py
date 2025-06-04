from saturation.saturation_routine import saturate_structure

if __name__ == "__main__":
    saturate_structure(filename="POSCAR_Siral_10", #initial filename, for the structures grown on top of other - POSCAR file is needed
                       mace_model_path="2024-01-07-mace-128-L2_epoch-199.model", #path to the MACE-MP model
                       device="mps") #device to calculate on
