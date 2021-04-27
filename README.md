# DFT Data: Influence of Br-/S2- site-exchange on Li diffusion mechanism in Li6PS5Br - a computational study

This repository contains results related to the publication:

Sadowski. M, Albe K., "Influence of Br-/S2- site-exchange on Li diffusion mechanism in Li6PS5Br - a computational study" Philosophical Transaction A,
DOI: https://doi.org/10.1098/rsta.2019.0458

The data has been obtained by running ab-initio molecular dynamics simulations using VASP (https://www.vasp.at/). The vasp code and its pseudopotentials are proprietary and cannot be made public. All remaining input files needed to repeat the calculations (POSCAR, INCAR, KPOINTS) can be found in the corresponding paths. Additionally, the obtained XDATCAR files that contain the atomic trajectories are made available here. 

Because one run on our computing cluster (limited to 24h run time) has not been enough to gather sufficient data we needed to restart the calculations several times. The XDATCARs of the individual runs have therefore been concatenated to generate the XDATCARs that are made available here. Note, that the XDATCARs for the Br_S and the S_Br defects contain every computed timestep, whereas the XDATCARs of the structures without anion defects (here: data/ordered/) and with an anion defect pair (here: data/6.25%disorder) only contain every 5th timestep.
