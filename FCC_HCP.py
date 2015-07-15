
from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *


# Set up the crystal structure
atomic_spacing = 3.78

fcc_lattice = FCC(atomic_spacing*np.sqrt(2))#0.52)
fcc_basis = Basis([('H',[0,0,0])])
fcc_crystal = fcc_lattice + fcc_basis


hcp_lattice = Hexagonal(atomic_spacing, atomic_spacing*np.sqrt(8/3))
hcp_basis = Basis([('H',[0,0,0]),
                   ('H', [0.5,0.5/np.sqrt(3),np.sqrt(2/3)])],
                  l_const=atomic_spacing)
hcp_crystal = hcp_lattice + hcp_basis


# Plot a simulated XRD with copper radiation
fcc_data = powder_XRD(fcc_crystal, 2.255)
fcc_angles, fcc_intensity = spectrumify(fcc_data)
hcp_data = powder_XRD(hcp_crystal, 2.255)
hcp_angles, hcp_intensity = spectrumify(hcp_data)
window = np.logical_and(fcc_angles<=55,fcc_angles>=25)


def calc_HCP_percentage(peak_1,peak_2):
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
    fcc_scaled = fcc_scaling * fcc_intensity[window],
    hcp_scaled = hcp_scaling * hcp_intensity[window],
    return (fcc_angles[window], fcc_scaled, hcp_scaled,
            fcc_scaled + hcp_scaled)

if __name__ == '__main__':

    #for ang, inten in sorted(hcp_data.items()):
    #    print("%.2f" % ang, inten)

    p.plot(fcc_angles[window], fcc_intensity[window])
    p.plot(hcp_angles[window], hcp_intensity[window])
    
    # Add some more info to the plot
    p.title(r'Simulated Powder XRD of FCC and HCP Hydrogen at 5498 eV')
    p.xlabel(r'$2\theta$')
    p.ylabel(r'Scattering Intensity per Cubic Angstrom')
    p.legend(['FCC','HCP'])
    p.show()
