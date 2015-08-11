from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *
from time import time

data_folder = '/Users/abe/Desktop/H2 Jet/H2 Jet Data/'
exp_angles = np.loadtxt(data_folder + 'angles.csv',
                         delimiter=',')*180/np.pi
spectra_1 = np.loadtxt(data_folder + 'spectra_1.csv',delimiter=',')
spectra_2 = np.loadtxt(data_folder + 'spectra_2.csv',delimiter=',')
exp_spectra = np.hstack([spectra_1,spectra_2])
exp_spectrum = np.sum(exp_spectra,axis=1)
exp_window = (exp_angles < 52) & (exp_angles > 38)
p.plot(exp_angles[exp_window],(exp_spectrum[exp_window]-0.015)
       /max(exp_spectrum[exp_window]-0.015),'k')
p.xlabel(r'$2\theta$')
p.ylabel('Normalized Intensity')
p.show()


atomic_spacing = 3.782
wavelength = 2.253

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
eigenmode_3 = np.array([[-0.70710678,  0.        ,  0.        ],
                        [ 0.        ,  0.70710678,  0.        ],
                        [ 0.        ,  0.        ,  0.        ]])


#6 - higher energy
eigenmode_4 = np.array([[ 0.        ,  0.70710678,  0.        ],
                        [ 0.70710678,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ]])


I = np.eye(3)


a = atomic_spacing
c = atomic_spacing*np.sqrt(8/3)
hcp_lattices = []
hcp_bases = []
hcp_crystals = []
hcp_factors_1 = []
hcp_factors_2 = []
hcp_factors_3 = []
for strain_1 in [0,0.03,0.06]:#np.linspace(0,0.08,20):
    for strain_2 in [0]:#np.linspace(0,0.08,20):
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
# Simulate the powder XRD
#

fcc_data = powder_XRD(fcc_crystal, 2.253)
fcc_angles, fcc_intensity = spectrumify(fcc_data)
hcp_data_list = [powder_XRD(hcp_crystal, 2.253) 
                 for hcp_crystal in hcp_crystals]
hcp_angles, hcp_intensities = zip(*[spectrumify(hcp_data) 
                                  for hcp_data in hcp_data_list])
hcp_angles = hcp_angles[0]
window = np.logical_and(fcc_angles<=52,fcc_angles>=38)

# hcp_intensity_1 = sum([hcp_intensity*hcp_factor
#                      for hcp_intensity, hcp_factor
#                        in zip(hcp_intensities,hcp_factors_1)]) / sum(hcp_factors_1)
# hcp_intensity_2 = sum([hcp_intensity*hcp_factor
#                      for hcp_intensity, hcp_factor
#                        in zip(hcp_intensities,hcp_factors_2)]) / sum(hcp_factors_2)
# hcp_intensity_3 = sum([hcp_intensity*hcp_factor
#                      for hcp_intensity, hcp_factor
#                        in zip(hcp_intensities,hcp_factors_3)]) / sum(hcp_factors_3)
# comb_intensity_1 = 0.35*fcc_intensity + 0.65*hcp_intensity_1
# comb_intensity_2 = 0.35*fcc_intensity + 0.65*hcp_intensity_2
# comb_intensity_3 = 0.35*fcc_intensity + 0.65*hcp_intensity_3

# p.plot(fcc_angles[window], comb_intensity_1[window] / max(comb_intensity_1[window]))
# p.plot(fcc_angles[window], comb_intensity_2[window] / max(comb_intensity_2[window]))
# p.plot(fcc_angles[window], comb_intensity_3[window] / max(comb_intensity_3[window]))
# p.plot(exp_angles[exp_window],(exp_spectrum[exp_window]-0.015)
#        /max(exp_spectrum[exp_window]-0.015),'k')

for hcp_intensity in hcp_intensities:
    p.plot(hcp_angles[window], hcp_intensity[window])
# Add some more info to the plot
#p.title(r'Simulated Spectra of HCP Hydrogen under Mode 1 Strain')
p.xlabel(r'$2\theta$')
p.ylabel(r'Normalized Intensity')
#p.legend(['1% strain','2% strain','3% strain','Real Data'])
p.legend(['No strain','3% strain','6% strain'])
p.show()
