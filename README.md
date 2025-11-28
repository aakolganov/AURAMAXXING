AURAMAXXING (Amorphous sUrface Research And Modeling And oXide eXploration Integrated iN Generation)

This repository provides tools to grow and saturate mixed Si/Al (at the moment) amorphous oxide structures using ASE and LAMMPS (for growth) and ASE + MACE (for saturation). 

Main Contributors:

- [Alexander Kolganov](https://github.com/aakolganov)
- [Mas Klein](https://github.com/MasKlein-1)


Installation & Dependencies

1. Python 3.9+
2. ASE
```bash
pip install ase
```
3. NumPy
```bash
pip install numpy
```
4. TQDM
```bash
pip install tqdm
```
5. LAMMPS (https://lammps.org). Use either the pip install below or install the lammps python library from source at https://github.com/lammps/lammps
```bash
pip install lammps
```

6. PyTorch
```bash
pip install torch
```
7. MACE (https://github.com/ACEsuit/mace)
```bash
pip install mace-torch
```
The representative examples script to grow and saturate a Si/Al oxide structure is provided in the folder "examples".
