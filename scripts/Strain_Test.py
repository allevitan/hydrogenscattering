from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *

atomic_spacing = 3.782

fcc_lattice = FCC(atomic_spacing*np.sqrt(2))
fcc_basis = Basis([('H',[0,0,0])])
fcc_crystal = fcc_lattice + fcc_basis

#3
eigenmode_1 = np.array([[ 0.        ,  0.        ,  0.39800322],
                      [ 0.        ,  0.        , -0.58445996],
                      [ 0.39800322, -0.58445996,  0.        ]])

#4
eigenmode_2 = np.array([[ 0.        ,  0.        , -0.58445996],
                      [ 0.        ,  0.        , -0.39800322],
                      [-0.58445996, -0.39800322,  0.        ]])

#5 - higher energy
#eigenmode = np.array([[-0.70710678,  0.        ,  0.        ],
#                      [ 0.        ,  0.70710678,  0.        ],
#                      [ 0.        ,  0.        ,  0.        ]])


#6 - higher energy
#eigenmode = np.array([[ 0.        ,  0.70710678,  0.        ],
#                      [ 0.70710678,  0.        ,  0.        ],
#                      [ 0.        ,  0.        ,  0.        ]])


I = np.eye(3)


a = atomic_spacing
c = atomic_spacing*np.sqrt(8/3)
hcp_lattices = []
hcp_bases = []
hcp_crystals = []
for strain_1 in [0]:#np.linspace(0,0.05,10):
    for strain_2 in [0]:#np.linspace(0,0.05,10):
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

#
# Simulate the powder XRD
#

fcc_data = powder_XRD(fcc_crystal, 2.265)
fcc_angles, fcc_intensity = spectrumify(fcc_data)
hcp_data_list = [powder_XRD(hcp_crystal, 2.265) 
                 for hcp_crystal in hcp_crystals]
hcp_angles, hcp_intensities = zip(*[spectrumify(hcp_data) 
                                  for hcp_data in hcp_data_list])
hcp_intensity = sum(hcp_intensities) / len(hcp_intensities)
hcp_angles = hcp_angles[0]
window = np.logical_and(fcc_angles<=55,fcc_angles>=35)

p.plot(fcc_angles[window], fcc_intensity[window])
p.plot(hcp_angles[window], hcp_intensity[window])
# Add some more info to the plot
p.title(r'Simulated Powder XRD of FCC and HCP Hydrogen at 5498 eV')
p.xlabel(r'$2\theta$')
p.ylabel(r'Scattering Intensity per Cubic Angstrom')
p.legend(['FCC','HCP'])
p.show()
