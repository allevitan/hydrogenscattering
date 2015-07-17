from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *

#
# This file contains functions that perform calculations with the
# "ideal" powder XRD spectra - both generating fake spectra for
# ideal mixtures, and using real spectra to calculate the volume
# fractions of fcc and hcp phases using the ideal spectrum.
#
# It is worth noting that this file is very specialized - it's not
# very portable and wouldn't work on other scattering systems
#


#
# Set up the crystal structures
#

atomic_spacing = 3.782

fcc_lattice = FCC(atomic_spacing*np.sqrt(2))
fcc_basis = Basis([('H',[0,0,0])])
fcc_crystal = fcc_lattice + fcc_basis


hcp_lattice = Hexagonal(atomic_spacing, atomic_spacing*np.sqrt(8/3))
hcp_basis = Basis([('H',[0,0,0]),
                   ('H', [0.5,0.5/np.sqrt(3),np.sqrt(2/3)])],
                  l_const=atomic_spacing)
hcp_crystal = hcp_lattice + hcp_basis


#
# Simulate the powder XRD
#

fcc_data = powder_XRD(fcc_crystal, 2.265)
fcc_angles, fcc_intensity = spectrumify(fcc_data)
hcp_data = powder_XRD(hcp_crystal, 2.265)
hcp_angles, hcp_intensity = spectrumify(hcp_data)
window = np.logical_and(fcc_angles<=55,fcc_angles>=35)


#
# Define some functions to play with that data
#

def calc_HCP_percentage(peak_1,peak_2):
    """
    Takes the heights of the 40 degree peak and the 42 degree
    peak and turns that information into the volume percentage
    of HCP using the ideal spectra for the hcp and fcc phases"""
    hcp1 = sorted(hcp_data.items())[0][1]
    hcp2 = sorted(hcp_data.items())[1][1]
    fcc = sorted(fcc_data.items())[0][1]
    hcp_part = peak_1 / hcp1
    fcc_part = (peak_2 - peak_1 * hcp2 / hcp1) / fcc 
    return hcp_part / (hcp_part + fcc_part) * 100

def mixture_data(ratio):
    """
    Returns the raw angle:intensity data for any hcp/fcc ratio"""
    fcc_scaling = 1 / (ratio + 1)
    hcp_scaling = ratio / (ratio + 1)
    fcc_scaled = {angle: intensity * fcc_scaling 
                  for angle, intensity in fcc_data.items()}
    hcp_scaled = {angle: intensity * hcp_scaling 
                  for angle, intensity in hcp_data.items()}
    return fcc_scaled, hcp_scaled

def mixture_graph(ratio):
    """
    Returns data to plot an easily-viewable simulated spectrum
    with any provided hcp/fcc ratio"""
    fcc_scaling = 1 / (ratio + 1)
    hcp_scaling = ratio / (ratio + 1)
    fcc_scaled = fcc_scaling * fcc_intensity[window]
    hcp_scaled = hcp_scaling * hcp_intensity[window]
    return (fcc_angles[window], fcc_scaled, hcp_scaled,
            fcc_scaled + hcp_scaled)



if __name__ == '__main__':

    for ang, inten in sorted(fcc_data.items()):
        print("%.2f" % ang, inten / max(fcc_data.values()) * 100)

    p.plot(fcc_angles[window], fcc_intensity[window])
    p.plot(hcp_angles[window], hcp_intensity[window])
    
    # Add some more info to the plot
    p.title(r'Simulated Powder XRD of FCC and HCP Hydrogen at 5498 eV')
    p.xlabel(r'$2\theta$')
    p.ylabel(r'Scattering Intensity per Cubic Angstrom')
    p.legend(['FCC','HCP'])
    p.show()
