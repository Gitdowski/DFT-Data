# DFT Data: Influence of Br-/S2- site-exchange on Li diffusion mechanism in Li6PS5Br - a computational study

This repository contains results related to the publication:

Sadowski. M, Albe K., "Influence of Br-/S2- site-exchange on Li diffusion mechanism in Li6PS5Br - a computational study", Philosophical Transaction A,
DOI: https://doi.org/10.1098/rsta.2019.0458

The data has been obtained by running ab-initio molecular dynamics simulations using VASP (https://www.vasp.at/). The vasp code and its pseudopotentials are proprietary and cannot be made public. The POTCAR versions were PAW_PBE Li_sv 10Sep2004, PAW_PBE P 06Sep2000, PAW_PBE S 06Sep2000 and PAW_PBE Br 06Sep2000.

All remaining input files needed to repeat the calculations (POSCAR, INCAR, KPOINTS) can be found in the corresponding paths. Additionally, the obtained XDATCAR files that contain the atomic trajectories are made available here. 

Because one run on our computing cluster (limited to 24h run time) has not been enough to gather sufficient data we needed to restart the calculations several times. The XDATCARs of the individual runs have therefore been concatenated to generate the XDATCARs that are made available here. Note, that the XDATCARs for the Br_S and the S_Br defects contain every computed timestep, whereas the XDATCARs of the structures without anion defects (here: data/ordered/) and with an anion defect pair (here: data/6.25%disorder) only contain every 5th timestep.

OVITO Pro (https://www.ovito.org/) and its python interface has been used to analyze the XDATCAR files. The following two scripts have been used.

1. Li_density_ovito3.py: This script reads the XDATCAR files and computes time-averaged Li densities based on a user-defined grid. The output is an xyz file (Li_density.xyz) containing the coordinates of the grid point and the corresponding Li density. 
  
2. Li_tetra_type.py: This file uses the POSCAR file in order to determine the distribution of S and Br atoms in the structure first. Afterwards it reads the previously generated Li_density.xyz file and determines the Li occupation of tetrahedral T1, T2, T3, T4 and T5 sites. The results are found in the Tetrahdral_Occupancies_* files.
