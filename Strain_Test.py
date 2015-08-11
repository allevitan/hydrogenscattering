from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *
from time import time


#
#
# This script uses calculated strain modes of HCP to simulate
# the effect of strain in the low energy modes on the XRD spectrum.
# It both calculates several examples of uniform strain and some
# examples of averaged strain 
#
#



# Set up the basic parameters for the simulation
atomic_spacing = 3.782
wavelength = 2.253

# And set up the FCC lattice - not gonna mess with this one
fcc_lattice = FCC(atomic_spacing*np.sqrt(2))
fcc_basis = Basis([('H',[0,0,0])])
fcc_crystal = fcc_lattice + fcc_basis
fcc_data = powder_XRD(fcc_crystal, 2.253)
fcc_angles, fcc_intensity = spectrumify(fcc_data)



# The two low energy eigenmodes of strain
eigenmode_1 = np.array([[ 0.        ,  0.        ,  0.39800322],
                      [ 0.        ,  0.        , -0.58445996],
                      [ 0.39800322, -0.58445996,  0.        ]])
eigenmode_2 = np.array([[ 0.        ,  0.        , -0.58445996],
                      [ 0.        ,  0.        , -0.39800322],
                      [-0.58445996, -0.39800322,  0.        ]])


# The two next-lowest-energy eigenmodes
eigenmode_3 = np.array([[-0.70710678,  0.        ,  0.        ],
                        [ 0.        ,  0.70710678,  0.        ],
                        [ 0.        ,  0.        ,  0.        ]])
eigenmode_4 = np.array([[ 0.        ,  0.70710678,  0.        ],
                        [ 0.70710678,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ]])


I = np.eye(3)



#
# We start by showing the effect that a single strain mode has
# on the spectrum
#


# First set up all the crystals
a = atomic_spacing
c = atomic_spacing*np.sqrt(8/3)
hcp_lattices = []
hcp_bases = []
hcp_crystals = []
for strain_1 in [0,0.03,0.06]:
    epsilon = I + strain_1 * eigenmode_1
    l = Lattice(a*np.dot(epsilon,n.array([1,0,0])),
                a*np.dot(epsilon,n.array([-0.5,n.sqrt(3)/2,0])),
                c*np.dot(epsilon,n.array([0,0,1])))
    b = Basis([('H',[0,0,0]),
               ('H', l.lattice[0]*1/3 + 
                l.lattice[1]*2/3 + 
                l.lattice[2]*0.5)])
    hcp_lattices.append(l)
    hcp_bases.append(b)
    hcp_crystals.append(l+b)


# Then simulate the powder XRD
hcp_data_list = [powder_XRD(hcp_crystal, 2.253) 
                 for hcp_crystal in hcp_crystals]
hcp_angles, hcp_intensities = zip(*[spectrumify(hcp_data) 
                                  for hcp_data in hcp_data_list])
hcp_angles = hcp_angles[0]
window = np.logical_and(fcc_angles<=52,fcc_angles>=38)


# And plot the result
for hcp_intensity in hcp_intensities:
    p.plot(hcp_angles[window], hcp_intensity[window])

p.title(r'Simulated Spectra of HCP Hydrogen under Mode 1 Strain')
p.xlabel(r'$2\theta$')
p.ylabel(r'Normalized Intensity')
p.legend(['No strain','3% strain','6% strain'])
p.show()




#
# Now we look at the effect of many normally distributed strain modes
#


# First set up the crystals 
a = atomic_spacing
c = atomic_spacing*np.sqrt(8/3)
hcp_lattices = []
hcp_bases = []
hcp_crystals = []
hcp_factors_1 = []
hcp_factors_2 = []
hcp_factors_3 = []
for strain_1 in np.linspace(0,0.08,20):
    for strain_2 in np.linspace(0,0.08,20):
        epsilon = I + strain_1 * eigenmode_1 + strain_2 * eigenmode_2
        l = Lattice(a*np.dot(epsilon,n.array([1,0,0])),
                    a*np.dot(epsilon,n.array([-0.5,n.sqrt(3)/2,0])),
                    c*np.dot(epsilon,n.array([0,0,1])))
        b = Basis([('H',[0,0,0]),
                   ('H', l.lattice[0]*1/3 + 
                    l.lattice[1]*2/3 + 
                    l.lattice[2]*0.5)])
        hcp_lattices.append(l)
        hcp_bases.append(b)
        hcp_crystals.append(l+b)
        hcp_factors_1.append(np.exp(-(strain_1**2 + strain_2**2)/(2*0.01**2)))
        hcp_factors_2.append(np.exp(-(strain_1**2 + strain_2**2)/(2*0.02**2)))
        hcp_factors_3.append(np.exp(-(strain_1**2 + strain_2**2)/(2*0.03**2)))

#
# Then simulate the powder XRD
#

hcp_data_list = [powder_XRD(hcp_crystal, 2.253) 
                 for hcp_crystal in hcp_crystals]
hcp_angles, hcp_intensities = zip(*[spectrumify(hcp_data) 
                                  for hcp_data in hcp_data_list])
hcp_angles = hcp_angles[0]
window = np.logical_and(fcc_angles<=52,fcc_angles>=38)

hcp_intensity_1 = sum([hcp_intensity*hcp_factor
                     for hcp_intensity, hcp_factor
                       in zip(hcp_intensities,hcp_factors_1)]) / sum(hcp_factors_1)
hcp_intensity_2 = sum([hcp_intensity*hcp_factor
                     for hcp_intensity, hcp_factor
                       in zip(hcp_intensities,hcp_factors_2)]) / sum(hcp_factors_2)
hcp_intensity_3 = sum([hcp_intensity*hcp_factor
                     for hcp_intensity, hcp_factor
                       in zip(hcp_intensities,hcp_factors_3)]) / sum(hcp_factors_3)
comb_intensity_1 = 0.35*fcc_intensity + 0.65*hcp_intensity_1
comb_intensity_2 = 0.35*fcc_intensity + 0.65*hcp_intensity_2
comb_intensity_3 = 0.35*fcc_intensity + 0.65*hcp_intensity_3


# And plot the result
p.plot(fcc_angles[window], comb_intensity_1[window] / max(comb_intensity_1[window]))
p.plot(fcc_angles[window], comb_intensity_2[window] / max(comb_intensity_2[window]))
p.plot(fcc_angles[window], comb_intensity_3[window] / max(comb_intensity_3[window]))
p.xlabel(r'$2\theta$')
p.ylabel(r'Normalized Intensity')
p.legend(['1% strain','2% strain','3% strain','Real Data'])
p.show()

