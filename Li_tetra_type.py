import timeit
tic=timeit.default_timer()

from ovito.io import import_file
from ovito.modifiers import *
from ovito.data import *
import numpy as np

# The Li density needs extended xyz format for the PBC to be set properly!
# Furthermore, a rather fine bin mesh is helpful to reduce numerical noise (because depending on the
# atomic positions some bins might just be located inside or outside the tetrahdral sites
Li_density_file="Li_density.xyz"

# The structure file is only used initially. Take an initial POSCAR without
# any displacements or, if not possible otherwise, the first frame of an XDATCAR.
# (With too large displacements the declaration of the 4d/4a sites might be screwed up)
Structure_file="POSCAR"




### Part 1: Take the initial POSCAR and determine how the S and Br are
#           distributed among the 4a and 4d sites. Their coordinates
#           are needed for the affine transformation of the Li_density.xyz
#           grid to see differences around the different coordinations.
#           S  on 4d site: ptype -> 10
#           Br on 4d site: ptype -> 11
#           S  on 4a site: ptype -> 12
#           Br on 4a site: ptype -> 13

# Sites that are usually occupied S in case of 0% disorder
Dict_4d_or_4c = {
"A1": [0.25, 0.0, 0.125],
"A2": [0.75, 0.0, 0.125],
"A3": [0.25, 0.5, 0.125],
"A4": [0.75, 0.5, 0.125],

"B1": [0.0, 0.25, 0.375],
"B2": [0.5, 0.25, 0.375],
"B3": [0.0, 0.75, 0.375],
"B4": [0.5, 0.75, 0.375],

"C1": [0.25, 0.0, 0.625],
"C2": [0.75, 0.0, 0.625],
"C3": [0.25, 0.5, 0.625],
"C4": [0.75, 0.5, 0.625],

"D1": [0.0, 0.25, 0.875],
"D2": [0.5, 0.25, 0.875],
"D3": [0.0, 0.75, 0.875],
"D4": [0.5, 0.75, 0.875]
}

# Sites that are usually occupied Br in case of 0% disorder
Dict_4a = {
"0.00-1": [0.25, 0.25, 0.0],
"0.00-2": [0.25, 0.75, 0.0],
"0.00-3": [0.75, 0.25, 0.0],
"0.00-4": [0.75, 0.75, 0.0],

"0.25-1": [0.0, 0.0, 0.25],
"0.25-2": [0.5, 0.0, 0.25],
"0.25-3": [0.0, 0.5, 0.25],
"0.25-4": [0.5, 0.5, 0.25],

"0.50-1": [0.25, 0.25, 0.5],
"0.50-2": [0.25, 0.75, 0.5],
"0.50-3": [0.75, 0.25, 0.5],
"0.50-4": [0.75, 0.75, 0.5],

"0.75-1": [0.0, 0.0, 0.75],
"0.75-2": [0.5, 0.0, 0.75],
"0.75-3": [0.0, 0.5, 0.75],
"0.75-4": [0.5, 0.5, 0.75]
}

# Load the simulation trajectory consisting of several frames.
# Pipeline1 only used for the initial analysis of sites:
pipeline1 = import_file(Structure_file)
data = pipeline1.compute()
SimBox=data.cell #used later to adapt Li density Box and to calculate the shifts
ptype = data.particles_["Particle Type_"]

# Store Atom IDs of relevant atoms
Li_id = ptype.type_by_name('Li').id
Br_id = ptype.type_by_name('Br').id
try:
    S_id = ptype.type_by_name('S2').id
except KeyError:
    S_id = ptype.type_by_name('S').id

# Create Nearest Neighbor Finder
N = 1
nearest_finder = NearestNeighborFinder(N, data)

# Change particle type of S/Br on 4d sites
for coord in Dict_4d_or_4c.values():
    x = coord[0] * data.cell[0][0]
    y = coord[1] * data.cell[1][1]
    z = coord[2] * data.cell[2][2]
    # Use Neighborfinder find_at() function to find closest atom to hard-coded 4d/4c coords
    closest_atom = [neigh.index for neigh in nearest_finder.find_at((x,y,z))][0]
    # S on 4d site: ptype -> 10
    if ptype[closest_atom] == S_id:
        ptype[closest_atom] = 10
    #Br on 4d site: ptype -> 11
    elif ptype[closest_atom] == Br_id:
        ptype[closest_atom] = 11
    elif ptype[closest_atom] == Li_id:
            print("Error! A Li was probably closer to the 4a site than the S or Br ion! check for id {}".format(closest_atom))

# Change particle type of S/Br on 4a sites
for coord in Dict_4a.values():
    x = coord[0]*data.cell[0][0]
    y = coord[1]*data.cell[1][1]
    z = coord[2]*data.cell[2][2]
    # Use Neighborfinder find_at() function to find closest atom to hard-coded 4d/4c coords
    closest_atom = [neigh.index for neigh in nearest_finder.find_at((x,y,z))][0]
    # S on 4a site: ptype -> 12
    if ptype[closest_atom] == S_id:
        ptype[closest_atom] = 12
    #Br on 4a site: ptype -> 13
    elif ptype[closest_atom] == Br_id:
        ptype[closest_atom] = 13
    elif ptype[closest_atom] == Li_id:
            print("Error! A Li was probably closer to the 4a site than the S or Br ion! check for id {}".format(closest_atom))

# Go through all atoms again and store their coordinates in the respective list
S_on_4d_type_10  = []
Br_on_4d_type_11 = []
S_on_4a_type_12  = []
Br_on_4a_type_13 = []
for atom,Type in enumerate(data.particles["Particle Type"]):
    if data.particles["Particle Type"][atom] == 10:
        pos = [float(data.particles["Position"][atom][0]),
               float(data.particles["Position"][atom][1]),
               float(data.particles["Position"][atom][2])]
        S_on_4d_type_10.append(pos)
    if data.particles["Particle Type"][atom] == 11:
        pos = [float(data.particles["Position"][atom][0]),
               float(data.particles["Position"][atom][1]),
               float(data.particles["Position"][atom][2])]
        Br_on_4d_type_11.append(pos)
    if data.particles["Particle Type"][atom] == 12:
        pos = [float(data.particles["Position"][atom][0]),
               float(data.particles["Position"][atom][1]),
               float(data.particles["Position"][atom][2])]
        S_on_4a_type_12.append(pos)
    if data.particles["Particle Type"][atom] == 13:
        pos = [float(data.particles["Position"][atom][0]),
               float(data.particles["Position"][atom][1]),
               float(data.particles["Position"][atom][2])]
        Br_on_4a_type_13.append(pos)


print("\nPositional analysis:")
if (len(S_on_4d_type_10) == len(Br_on_4a_type_13)) and (len(S_on_4a_type_12) == len(Br_on_4d_type_11)):
    print("  Number comparison: S_Br=Br_S and S_S=Br_Br, therefore stoichiometric compound assumed")
    SiteDisorder = 100*len(S_on_4a_type_12)/(len(S_on_4a_type_12)+len(S_on_4d_type_10))
    print("  Site-disorder: {:.2f}% \n".format(SiteDisorder))

else:
    print("  Br and S anion sublattices not equally occupied: Non-stoichiometric compound or with defects.\n")
    SiteDisorder = -1


# Initial words: There are five different tetrahedral sites
# Type 1: only around 4d sites        -> evaluation based on the respective 4d site. Look at S vs Br
# Type 2: between 4a and 4d sites     -> makes sense to compare S and Br but also based on 4d and 4a site
# Type 3: Only between PS4^3- units   -> never occupied anyway (?)
# Type 4: only around 4a sites        -> evaluation based on the respective 4a site. Look at S vs Br
# Type 5: between 4a and 4d sites     -> makes sense to compare S and Br but also based on 4d and 4a site


### Part 2: a) First, define all tetrahedra based on their corners
#              (This has been done previously by constructing coordination polyhedra via dummy atoms and bonds between them
#              and the needed real atoms in the ovito GUI. Exporting it as vtk delivers the needed corners that can be put
#              into a list here.)
#
#           b) import Li_density
#           c) Remove all points with zero density to keep it clean
#           d) not needed anymore due to extended xyz format.
#              (previously: affine transformation to increase the box and make it fit to the POSCAR + enable PBC)
#
#           for all the different sites:
#           e) affine transformation to shift the site of interest into the center of the box + wrap density
#           f) python modifier to select only density points within tetrahedra and sum them up
#
#           g) collect results and export to file


# a) define tetrahedra
tetra_list_4d_type1 = np.array([[[7.2729001045, 7.2729000523, 10.2854474863],
                                 [8.9527950216, 10.9093501045, 11.6690458064],
                                 [5.5930056529, 10.9093501045, 11.6690458064],
                                 [7.2729001045, 9.2294556529, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [8.9527950216, 3.63645, 11.6690458064],
                                 [5.5930056529, 3.63645, 11.6690458064],
                                 [7.2729001045, 5.3163444516, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [10.9093503895, 5.5930054843, 8.9018486725],
                                 [10.9093503895, 8.9527942566, 8.9018486725],
                                 [9.2294549924, 7.2729000523, 6.5263210624]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [3.6364500523, 5.5930054843, 8.9018486725],
                                 [3.6364500523, 8.9527942566, 8.9018486725],
                                 [5.3163442566, 7.2729000523, 6.5263210624]]])

tetra_list_4d_type2 = np.array([[[7.2729001045, 7.2729000523, 10.2854474863],
                                 [10.9093503895, 7.2729000523, 12.8568099817],
                                 [8.9527950216, 10.9093501045, 11.6690458064],
                                 [7.2729001045, 9.2294556529, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [10.9093503895, 7.2729000523, 12.8568099817],
                                 [8.9527950216, 3.63645, 11.6690458064],
                                 [7.2729001045, 5.3163444516, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [10.9093503895, 7.2729000523, 12.8568099817],
                                 [10.9093503895, 5.5930054843, 8.9018486725],
                                 [10.9093503895, 8.9527942566, 8.9018486725]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [3.6364500523, 7.2729000523, 12.8568099817],
                                 [5.5930056529, 10.9093501045, 11.6690458064],
                                 [7.2729001045, 9.2294556529, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [3.6364500523, 7.2729000523, 12.8568099817],
                                 [5.5930056529, 3.63645, 11.6690458064],
                                 [7.2729001045, 5.3163444516, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [3.6364500523, 7.2729000523, 12.8568099817],
                                 [3.6364500523, 5.5930054843, 8.9018486725],
                                 [3.6364500523, 8.9527942566, 8.9018486725]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 10.9093501045, 7.7140849909],
                                 [10.9093503895, 8.9527942566, 8.9018486725],
                                 [9.2294549924, 7.2729000523, 6.5263210624]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 10.9093501045, 7.7140849909],
                                 [3.6364500523, 8.9527942566, 8.9018486725],
                                 [5.3163442566, 7.2729000523, 6.5263210624]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 10.9093501045, 7.7140849909],
                                 [8.9527950216, 10.9093501045, 11.6690458064],
                                 [5.5930056529, 10.9093501045, 11.6690458064]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 3.63645, 7.7140849909],
                                 [10.9093503895, 5.5930054843, 8.9018486725],
                                 [9.2294549924, 7.2729000523, 6.5263210624]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 3.63645, 7.7140849909],
                                 [3.6364500523, 5.5930054843, 8.9018486725],
                                 [5.3163442566, 7.2729000523, 6.5263210624]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 3.63645, 7.7140849909],
                                 [8.9527950216, 3.63645, 11.6690458064],
                                 [5.5930056529, 3.63645, 11.6690458064]]])

tetra_list_4a_type2 = np.array([[[10.9093503895, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [9.2294549924, 10.9093503895, 9.0976910624],
                                 [10.9093503895, 9.2294549924, 11.4732186725]],
                                [[10.9093503895, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [9.2294549924, 3.6364500523, 9.0976910624],
                                 [10.9093503895, 5.3163442566, 11.4732186725]],
                                [[3.6364500523, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [5.3163442566, 10.9093503895, 9.0976910624],
                                 [3.6364500523, 9.2294549924, 11.4732186725]],
                                [[3.6364500523, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [5.3163442566, 3.6364500523, 9.0976910624],
                                 [3.6364500523, 5.3163442566, 11.4732186725]],
                                [[10.9093503895, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 8.9527950216, 6.3304936816],
                                 [7.2729001045, 5.5930056529, 6.3304936816]],
                                [[3.6364500523, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 8.9527950216, 6.3304936816],
                                 [7.2729001045, 5.5930056529, 6.3304936816]],
                                [[7.2729001045, 10.9093503895, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [10.9093503895, 9.2294549924, 11.4732186725],
                                 [9.2294549924, 10.9093503895, 9.0976910624]],
                                [[7.2729001045, 3.6364500523, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [10.9093503895, 5.3163442566, 11.4732186725],
                                 [9.2294549924, 3.6364500523, 9.0976910624]],
                                [[7.2729001045, 10.9093503895, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [3.6364500523, 9.2294549924, 11.4732186725],
                                 [5.3163442566, 10.9093503895, 9.0976910624]],
                                [[7.2729001045, 3.6364500523, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [3.6364500523, 5.3163442566, 11.4732186725],
                                 [5.3163442566, 3.6364500523, 9.0976910624]],
                                [[7.2729001045, 10.9093503895, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [8.9527950216, 7.2729001045, 14.2404158064],
                                 [5.5930056529, 7.2729001045, 14.2404158064]],
                                [[7.2729001045, 3.6364500523, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [8.9527950216, 7.2729001045, 14.2404158064],
                                 [5.5930056529, 7.2729001045, 14.2404158064]]])

tetra_list_4d_type3 = np.array([[[5.3163442566, 7.2729000523, 16.8117717641],
                                 [7.2729001045, 5.3163444516, 14.0445731902],
                                 [7.2729001045, 9.2294556529, 14.0445731902],
                                 [9.2294549924, 7.2729000523, 16.8117717641]]])

tetra_list_4d_type4 = np.array([[[9.2294549924, 7.2729000523, 16.8117717641],
                                 [10.9093503895, 7.2729000523, 12.8568099817],
                                 [7.2729001045, 5.3163444516, 14.0445731902],
                                 [7.2729001045, 9.2294556529, 14.0445731902]],
                                [[5.3163442566, 7.2729000523, 16.8117717641],
                                 [3.6364500523, 7.2729000523, 12.8568099817],
                                 [7.2729001045, 5.3163444516, 14.0445731902],
                                 [7.2729001045, 9.2294556529, 14.0445731902]],
                                [[5.3163442566, 7.2729000523, 6.5263210624],
                                 [7.2729001045, 10.9093501045, 7.7140849909],
                                 [7.2729001045, 9.2294556529, 3.7591236816],
                                 [9.2294549924, 7.2729000523, 6.5263210624]],
                                [[5.3163442566, 7.2729000523, 6.5263210624],
                                 [7.2729001045, 3.63645, 7.7140849909],
                                 [7.2729001045, 5.3163444516, 3.7591236816],
                                 [9.2294549924, 7.2729000523, 6.5263210624]]])

tetra_list_4a_type4 = np.array([[[10.9093503895, 9.2294549924, 11.4732186725],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [8.9527950216, 7.2729001045, 14.2404158064],
                                 [10.9093503895, 5.3163442566, 11.4732186725]],
                                [[3.6364500523, 5.3163442566, 11.4732186725],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [5.5930056529, 7.2729001045, 14.2404158064],
                                 [3.6364500523, 9.2294549924, 11.4732186725]],
                                [[9.2294549924, 10.9093503895, 9.0976910624],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 8.9527950216, 6.3304936816],
                                 [5.3163442566, 10.9093503895, 9.0976910624]],
                                [[5.3163442566, 3.6364500523, 9.0976910624],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 5.5930056529, 6.3304936816],
                                 [9.2294549924, 3.6364500523, 9.0976910624]]])

tetra_list_4d_type5 = np.array([[[7.2729001045, 7.2729000523, 10.2854474863],
                                 [10.9093503895, 7.2729000523, 12.8568099817],
                                 [7.2729001045, 5.3163444516, 14.0445731902],
                                 [7.2729001045, 9.2294556529, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [3.6364500523, 7.2729000523, 12.8568099817],
                                 [7.2729001045, 5.3163444516, 14.0445731902],
                                 [7.2729001045, 9.2294556529, 14.0445731902]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 10.9093501045, 7.7140849909],
                                 [5.3163442566, 7.2729000523, 6.5263210624],
                                 [9.2294549924, 7.2729000523, 6.5263210624]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 3.63645, 7.7140849909],
                                 [5.3163442566, 7.2729000523, 6.5263210624],
                                 [9.2294549924, 7.2729000523, 6.5263210624]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [10.9093503895, 7.2729000523, 12.8568099817],
                                 [10.9093503895, 5.5930054843, 8.9018486725],
                                 [8.9527950216, 3.63645, 11.6690458064]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 3.63645, 7.7140849909],
                                 [8.9527950216, 3.63645, 11.6690458064],
                                 [10.9093503895, 5.5930054843, 8.9018486725]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [10.9093503895, 7.2729000523, 12.8568099817],
                                 [10.9093503895, 8.9527942566, 8.9018486725],
                                 [8.9527950216, 10.9093501045, 11.6690458064]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 10.9093501045, 7.7140849909],
                                 [8.9527950216, 10.9093501045, 11.6690458064],
                                 [10.9093503895, 8.9527942566, 8.9018486725]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [3.6364500523, 7.2729000523, 12.8568099817],
                                 [3.6364500523, 5.5930054843, 8.9018486725],
                                 [5.5930056529, 3.63645, 11.6690458064]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 3.63645, 7.7140849909],
                                 [5.5930056529, 3.63645, 11.6690458064],
                                 [3.6364500523, 5.5930054843, 8.9018486725]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [3.6364500523, 7.2729000523, 12.8568099817],
                                 [3.6364500523, 8.9527942566, 8.9018486725],
                                 [5.5930056529, 10.9093501045, 11.6690458064]],
                                [[7.2729001045, 7.2729000523, 10.2854474863],
                                 [7.2729001045, 10.9093501045, 7.7140849909],
                                 [5.5930056529, 10.9093501045, 11.6690458064],
                                 [3.6364500523, 8.9527942566, 8.9018486725]]])

tetra_list_4a_type5 = np.array([[[7.2729001045, 10.9093503895, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [8.9527950216, 7.2729001045, 14.2404158064],
                                 [10.9093503895, 9.2294549924, 11.4732186725]],
                                [[7.2729001045, 10.9093503895, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [5.5930056529, 7.2729001045, 14.2404158064],
                                 [3.6364500523, 9.2294549924, 11.4732186725]],
                                [[7.2729001045, 3.6364500523, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [8.9527950216, 7.2729001045, 14.2404158064],
                                 [10.9093503895, 5.3163442566, 11.4732186725]],
                                [[7.2729001045, 3.6364500523, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [5.5930056529, 7.2729001045, 14.2404158064],
                                 [3.6364500523, 5.3163442566, 11.4732186725]],
                                [[10.9093503895, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 8.9527950216, 6.3304936816],
                                 [9.2294549924, 10.9093503895, 9.0976910624]],
                                [[3.6364500523, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 8.9527950216, 6.3304936816],
                                 [5.3163442566, 10.9093503895, 9.0976910624]],
                                [[10.9093503895, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 5.5930056529, 6.3304936816],
                                 [9.2294549924, 3.6364500523, 9.0976910624]],
                                [[3.6364500523, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [7.2729001045, 5.5930056529, 6.3304936816],
                                 [5.3163442566, 3.6364500523, 9.0976910624]],
                                [[10.9093503895, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [10.9093503895, 5.3163442566, 11.4732186725],
                                 [10.9093503895, 9.2294549924, 11.4732186725]],
                                [[3.6364500523, 7.2729001045, 7.7140924954],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [3.6364500523, 5.3163442566, 11.4732186725],
                                 [3.6364500523, 9.2294549924, 11.4732186725]],
                                [[7.2729001045, 10.9093503895, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [5.3163442566, 10.9093503895, 9.0976910624],
                                 [9.2294549924, 10.9093503895, 9.0976910624]],
                                [[7.2729001045, 3.6364500523, 12.8568174863],
                                 [7.2729001045, 7.2729001045, 10.2854549908],
                                 [5.3163442566, 3.6364500523, 9.0976910624],
                                 [9.2294549924, 3.6364500523, 9.0976910624]]])



def TranslationToCenterOfBox(target_box,pos):
    """
    Takes the target box and determines the translation matrix
    for the affine transformation modifier
    returns translation matrix
    """
    x_shift = target_box[0][0] / 2 - pos[0]
    y_shift = target_box[1][1] / 2 - pos[1]
    z_shift = target_box[2][2] / 2 - pos[2]
    trans =[[1.0, 0.0, 0.0, x_shift],
            [0.0, 1.0, 0.0, y_shift],
            [0.0, 0.0, 1.0, z_shift]]
    return trans

#Tetrahedron and pointInside Taken from https://stackoverflow.com/a/60745339
def Tetrahedron(vertices):
    """
    Given a list of the xyz coordinates of the vertices of a tetrahedron,
    return tetrahedron coordinate system
    """
    origin, *rest = vertices
    mat = (np.array(rest) - origin).T
    tetra = np.linalg.inv(mat)
    return tetra, origin


def pointInside(point, tetra, origin):
    """
    Takes a single point or array of points, as well as tetra and origin objects returned by
    the Tetrahedron function.
    Returns a boolean or boolean array indicating whether the point is inside the tetrahedron.
    """
    newp = np.matmul(tetra, (point - origin).T).T
    return np.all(newp >= 0, axis=-1) & np.all(newp <= 1, axis=-1) & (np.sum(newp, axis=-1) <= 1)


def SelectDensityAndSumUp(frame: int, data: DataCollection, tetra_list):
    """
    Clears the selection first, then selects all points within the
    specified tetrahedra and sums up their densities
    returns summed density
    """
    # To be on the safe side: clear selection first
    sel = data.particles_["Selection_"]
    for i in range(len(sel)):
        sel[i]=0

    points = data.particles.positions

    for tetra in tetra_list:
        vertices = tetra
        tetra, origin = Tetrahedron(vertices)
        inTet = pointInside(points, tetra, origin)
        sel[inTet] = 1

        density = data.particles["Density"][data.particles["Selection"] == 1]

    return np.sum(density)

# b) import Li density and do some checks.
pipeline2=import_file(Li_density_file, columns =
  ["Position.X", "Position.Y", "Position.Z", "Density"])

data               = pipeline2.compute()
N_bins             = len(data.particles["Density"])
true_total_density = np.sum(data.particles["Density"])

print("\nNumber of bins:  {}".format(N_bins))
print("Total Density:   {:.3f}\n".format(true_total_density))


if (data.cell.pbc[0] == True) and (data.cell.pbc[1] == True) and (data.cell.pbc[2] == True):
    print("PBC are enabled. Continue...\n")
else:
    print("Ups! The PBC are not enabled. Make sure to use extended xyz format for the Li density file!\nAborting...")
    exit()


# c) remove all points with 0 Li density to deal with less data
pipeline2.modifiers.append(ExpressionSelectionModifier(expression = 'Density == 0'))
pipeline2.modifiers.append(DeleteSelectedModifier())

# Outdated because of extended xyz format:
# d) affine transformation to increase (only) the box and make it fit to the POSCAR + enable PBC
#pipeline2.modifiers.append(AffineTransformationModifier(
#    relative_mode = False,
#    operate_on = {'cell'},
#    target_cell = SimBox))#SimBox taken from StructureFile above
#pipeline2.modifiers.append(EnablePBC(data=data))

# e) Setup affine transformation modifier to shift the density + enable Wrapping
#    The shift itself will be specified in the for loop below based on the S/Br positions on 4d/4a
affine_translation_density_mod = AffineTransformationModifier(
    relative_mode = True,
    operate_on = {'particles'})
pipeline2.modifiers.append(affine_translation_density_mod)
pipeline2.modifiers.append(WrapPeriodicImagesModifier())

# f) python modifier to select only density points within tetrahedra and sum them up
total_density         = [0, 0, 0, 0, 0] #for type 1 to type 5

total_density_S_on_4d = [0, 0, float('nan'), float('nan'), 0]
if S_on_4d_type_10: #only do the analysis if list is not empty
    print("Started S on 4d...")
    count = 1
    for coordinate in S_on_4d_type_10:
        print("   ...progress: Atom {:d} of {:d}".format(count,len(S_on_4d_type_10)))
        # Shift the density to center of box
        Translation = TranslationToCenterOfBox(target_box=SimBox,pos=coordinate)
        affine_translation_density_mod.transformation = Translation
        # Execute the pipeline
        data = pipeline2.compute()
        data.particles_.create_property("Selection")
        # Does the actual analyis. Determines density:

        ### S on 4d type 1:
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type1)
        total_density_S_on_4d[0] = total_density_S_on_4d[0] + dens
        total_density[0] = total_density[0] + dens

        ### S on 4d type 2 (needs to analyzed based on S and Br as well as based from 4d and 4a):
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type2)
        total_density_S_on_4d[1] = total_density_S_on_4d[1] + dens
        total_density[1] = total_density[1] + dens

        ### S on 4d type 3 (should be zero anyway. No need to look at it based on 4a sites)
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type3)
        #total_density_S_on_4d[2] = total_density_S_on_4d[2] + dens
        total_density[2] = total_density[2] + dens

        ### type 4 is not around the 4d site. No need to evaluate it here for the total_density_S_on_4d, but needed for the total_density
        #dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type4)
        #total_density_S_on_4d[3] = total_density_S_on_4d[3] + dens
        #total_density[3] = total_density[3] + dens

        ### S on 4d type 5 (needs to analyzed based on S and Br as well as based from 4d and 4a):
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type5)
        total_density_S_on_4d[4] = total_density_S_on_4d[4] + dens
        total_density[4] = total_density[4] + dens

        count = count + 1

total_density_Br_on_4d = [0, 0, float('nan'), float('nan'), 0]
if Br_on_4d_type_11: #only do the analysis if list is not empty
    print("Started Br on 4d...")
    count = 1
    for coordinate in Br_on_4d_type_11:
        print("   ...progress: Atom {:d} of {:d}".format(count,len(Br_on_4d_type_11)))
        # Shift the density to center of box
        Translation = TranslationToCenterOfBox(target_box=SimBox,pos=coordinate)
        affine_translation_density_mod.transformation = Translation
        # Execute the pipeline
        data = pipeline2.compute()
        data.particles_.create_property("Selection")
        # Does the actual analyis. Determines density:

        ### Br on 4d type 1:
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type1)
        total_density_Br_on_4d[0] = total_density_Br_on_4d[0] + dens
        total_density[0] = total_density[0] + dens

        ### Br on 4d type 2 (needs to analyzed based on S and Br as well as based from 4d and 4a):
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type2)
        total_density_Br_on_4d[1] = total_density_Br_on_4d[1] + dens
        total_density[1] = total_density[1] + dens

        ### Br on 4d type 3 (should be zero anyway. No need to look at it based on 4a sites)
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type3)
        #total_density_Br_on_4d[2] = total_density_Br_on_4d[2] + dens
        total_density[2] = total_density[2] + dens

        ### type 4 is not around the 4d site. No need to evaluate it here for the total_density_Br_on_4d, but needed for the total_density
        #dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type4)
        #total_density_Br_on_4d[3] = total_density_Br_on_4d[3] + dens
        #total_density[3] = total_density[3] + dens

        ### Br on 4d type 5 (needs to analyzed based on S and Br as well as based from 4d and 4a):
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4d_type5)
        total_density_Br_on_4d[4] = total_density_Br_on_4d[4] + dens
        total_density[4] = total_density[4] + dens

        count = count + 1

total_density_S_on_4a = [float('nan'), 0, float('nan'), 0, 0]
if S_on_4a_type_12: #only do the analysis if list is not empty
    print("Started S on 4a...")
    count = 1
    for coordinate in S_on_4a_type_12:
        print("   ...progress: Atom {:d} of {:d}".format(count,len(S_on_4a_type_12)))
        # Shift the density to center of box
        Translation = TranslationToCenterOfBox(target_box=SimBox,pos=coordinate)
        affine_translation_density_mod.transformation = Translation
        # Execute the pipeline
        data = pipeline2.compute()
        data.particles_.create_property("Selection")
        # Does the actual analyis. Determines density:

        ### No type 1 site around the 4a site
        #dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type1)
        #total_density_S_on_4a[0] = total_density_Br_on_4d[0] + dens
        #total_density[0] = total_density[0] + dens

        ### Type 2 sites (around 4a and 4d sites)
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type2)
        total_density_S_on_4a[1] = total_density_S_on_4a[1] + dens
        total_density[1] = total_density[1] + dens

        ### Type 3 already evaluated based on the 4d sites

        ### Type 4 sites (only around 4a sites):
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type4)
        total_density_S_on_4a[3] = total_density_S_on_4a[3] + dens
        total_density[3] = total_density[3] + dens

        ### Type 5 site (around 4a and 4d sites)
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type5)
        total_density_S_on_4a[4] = total_density_S_on_4a[4] + dens
        total_density[4] = total_density[4] + dens

        count = count + 1

total_density_Br_on_4a = [float('nan'), 0, float('nan'), 0, 0]
if Br_on_4a_type_13: #only do the analysis if list is not empty
    print("Started Br on 4a...")
    count = 1
    for coordinate in Br_on_4a_type_13:
        print("   ...progress: Atom {:d} of {:d}".format(count,len(Br_on_4a_type_13)))
        # Shift the density to center of box
        Translation = TranslationToCenterOfBox(target_box=SimBox,pos=coordinate)
        affine_translation_density_mod.transformation = Translation
        # Execute the pipeline
        data = pipeline2.compute()
        data.particles_.create_property("Selection")
        # Does the actual analyis. Determines density:

        ### No type 1 site around the 4a site
        # dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type1)
        # total_density_S_on_4a[0] = total_density_Br_on_4d[0] + dens
        # total_density[0] = total_density[0] + dens

        ### Type 2 sites (around 4a and 4d sites)
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type2)
        total_density_Br_on_4a[1] = total_density_Br_on_4a[1] + dens
        total_density[1] = total_density[1] + dens

        ### Type 3 already evaluated based on the 4d sites

        ### Type 4 sites (only around 4a sites):
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type4)
        total_density_Br_on_4a[3] = total_density_Br_on_4a[3] + dens
        total_density[3] = total_density[3] + dens

        ### Type 5 site (around 4a and 4d sites)
        dens = SelectDensityAndSumUp(frame=0, data=data, tetra_list=tetra_list_4a_type5)
        total_density_Br_on_4a[4] = total_density_Br_on_4a[4] + dens
        total_density[4] = total_density[4] + dens

        count = count + 1
        

# g) Collect results and export

#Evaluaion of the average numerical density. Needed for a more appropriate normalization
#Type1 only around 4d; Type2 both around 4a and 4d; Type3 is zero anyway; Type4 only around 4a; Type5 around both
numerical_density=total_density[0]+total_density[1]/2+total_density[2]+total_density[3]+total_density[4]/2

print("\nNumerical Total Density   = {:.2f} (arbitrary units, not normalized)".format(numerical_density))
print("of the true Total Density = {:.2f} (arbitrary units, not normalized)\n".format(true_total_density))

n = numerical_density
t = total_density

with open("Tetrahdral_Occupancies_Percentages.txt", "w") as f:
    f.write("Site\tType1\tType2\tType3\tType4\tType5\t#So to say two different 'perspectives' (from 4a and 4d sites)\n")

    d1 = total_density_S_on_4d
    f.write("S_4d\t{:.3f}\t{:.3f}\t{}\t{}\t{:.3f}\n".format(100*d1[0]/n,100*d1[1]/n,".",".",100*d1[4]/n))

    d2 = total_density_Br_on_4d
    f.write("Br_4d\t{:.3f}\t{:.3f}\t{}\t{}\t{:.3f}\n".format(100*d2[0]/n,100*d2[1]/n,".",".",100*d2[4]/n))

    d3 = total_density_S_on_4a
    f.write("S_4a\t{}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\n".format(".",100*d3[1]/n,".",100*d3[3]/n,100*d3[4]/n))

    d4 = total_density_Br_on_4a
    f.write("Br_4a\t{}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\n".format(".",100*d4[1]/n,".",100*d4[3]/n,100*d4[4]/n))

    f.write("Total\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(100*t[0]/n, 100*t[1]/2/n, 100*t[2]/n, 100*t[3]/n, 100*t[4]/2/n))


with open("Tetrahdral_Occupancies_Percentages_reversed.txt", "w") as f:
    f.write("Type\tS_4d\tBr_4d\tS_4a\tBr_4a\tTotal\t#So to say two different 'perspectives' (from 4a and 4d sites)\n")

    dummy_types=[1,2,3,4,5]
    for Type,i,j,k,l,m in zip(dummy_types,total_density_S_on_4d,total_density_Br_on_4d,total_density_S_on_4a,total_density_Br_on_4a,t):
        f.write("Type{:d}\t".format(Type))

        if i >= 0.0:
            f.write("{:.3f}\t".format(100*i/n))
        else:
            f.write(".\t")

        if j >= 0.0:
            f.write("{:.3f}\t".format(100*j/n))
        else:
            f.write(".\t")

        if k >= 0.0:
            f.write("{:.3f}\t".format(100*k/n))
        else:
            f.write(".\t")

        if l >= 0.0:
            f.write("{:.3f}\t".format(100*l/n))
        else:
            f.write(".\t")

        if   Type == 1:
            f.write("{:.3f}\n".format(100*m/n))
        elif Type == 2:
            f.write("{:.3f}\n".format(100*m/2/n))
        elif Type == 3:
            f.write("{:.3f}\n".format(100*m/n))
        elif Type == 4:
            f.write("{:.3f}\n".format(100*m/n))
        elif Type == 5:
            f.write("{:.3f}\n".format(100*m/2/n))

###### Normalization is done per site.
# E.g. A value of 2.3 for type2 Li around S_4d means that on average 2.4 Li are distributed on the
# type2 tetrahedral sites around the S_4d sites.



# Case 1: Standard Site-Disorder, i.e. for every S_Br there is a Br_S
if not (SiteDisorder == -1):
    try:
        Norm1 = data.cell.volume/N_bins/len(S_on_4d_type_10)
        Norm4 = Norm1
    except ZeroDivisionError:
        Norm1 = 0.0
        Norm4 = 0.0

    try:
        Norm2 = data.cell.volume/N_bins/len(S_on_4a_type_12)
        Norm3 = Norm2
    except ZeroDivisionError:
        Norm2 = 0.0
        Norm3 = 0.0

# Case 2: No Standard Site-Disorder (i.e. defective systems) 
if (SiteDisorder == -1):
    try:
        Norm1 = data.cell.volume/N_bins/len(S_on_4d_type_10)
    except ZeroDivisionError:
        Norm1 = 0.0

    try:
        Norm2 = data.cell.volume/N_bins/len(Br_on_4d_type_11)
    except ZeroDivisionError:
        Norm2 = 0.0

    try:
        Norm3 = data.cell.volume/N_bins/len(S_on_4a_type_12)
    except ZeroDivisionError:
        Norm3 = 0.0

    try:
        Norm4 = data.cell.volume/N_bins/len(Br_on_4a_type_13)
    except ZeroDivisionError:
        Norm4 = 0.0


with open("Tetrahdral_Occupancies_Absolute_per_site.txt", "w") as f:
    f.write("Site\tType1\tType2\tType3\tType4\tType5\t#So to say two different 'perspectives' (from 4a and 4d sites)\n")

    d1 = total_density_S_on_4d
    f.write("S_4d\t{:.3f}\t{:.3f}\t{}\t{}\t{:.3f}\n".format(d1[0]*Norm1,d1[1]*Norm1,".",".",d1[4]*Norm1))

    d2 = total_density_Br_on_4d
    f.write("Br_4d\t{:.3f}\t{:.3f}\t{}\t{}\t{:.3f}\n".format(d2[0]*Norm2,d2[1]*Norm2,".",".",d2[4]*Norm2))

    d3 = total_density_S_on_4a
    f.write("S_4a\t{}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\n".format(".",d3[1]*Norm3,".",d3[3]*Norm3,d3[4]*Norm3))

    d4 = total_density_Br_on_4a
    f.write("Br_4a\t{}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\n".format(".",d4[1]*Norm4,".",d4[3]*Norm4,d4[4]*Norm4))




with open("Tetrahdral_Occupancies_Absolute_per_site_reversed.txt", "w") as f:
    f.write("Type\tS_4d\tBr_4d\tS_4a\tBr_4a\t#So to say two different 'perspectives' (from 4a and 4d sites)\n")

    dummy_types=[1,2,3,4,5]
    for Type,i,j,k,l in zip(dummy_types,total_density_S_on_4d,total_density_Br_on_4d,total_density_S_on_4a,total_density_Br_on_4a):
        f.write("Type{:d}\t".format(Type))

        if i >= 0.0:
            f.write("{:.3f}\t".format(i*Norm1))
        else:
            f.write(".\t")

        if j >= 0.0:
            f.write("{:.3f}\t".format(j*Norm2))
        else:
            f.write(".\t")

        if k >= 0.0:
            f.write("{:.3f}\t".format(k*Norm3))
        else:
            f.write(".\t")

        if l >= 0.0:
            f.write("{:.3f}\t".format(l*Norm4))
        else:
            f.write(".\t")

        f.write("\n")
