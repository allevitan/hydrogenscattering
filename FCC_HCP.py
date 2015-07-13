from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *


# Set up the crystal structure
atomic_spacing = 3.78

fcc_lattice = FCC(atomic_spacing*np.sqrt(2))#0.52)
fcc_basis = Basis([('H',[0,0,0])])
fcc_crystal = fcc_lattice + fcc_basis


a = atomic_spacing*np.array([1,0,0])
b = atomic_spacing*np.array([-0.5,np.sqrt(3)/2,0])
c = atomic_spacing*np.array([0,0,np.sqrt(8/3)])
hcp_lattice = Lattice(a,b,c)
hcp_basis = Basis([('H',[0,0,0]),
                   ('H',np.array([0.5,
                                  0.5/np.sqrt(3),
                                  np.sqrt(2/3)])*atomic_spacing)])
hcp_crystal = hcp_lattice + hcp_basis


# Plot a simulated XRD with copper radiation
fcc_data = powder_XRD(fcc_crystal, 2.255)
fcc_angles, fcc_intensity = spectrumify(fcc_data)
hcp_data = powder_XRD(hcp_crystal, 2.255)
hcp_angles, hcp_intensity = spectrumify(hcp_data)
window = np.logical_and(fcc_angles<=55,fcc_angles>=25)

def mixture_data(ratio):
    """
    Returns the raw angle:intensity data for any fcc/hcp ratio"""
    fcc_scaling = ratio / (ratio + 1)
    hcp_scaling = 1 / (ratio + 1)
    fcc_scaled = {angle: intensity * fcc_scaling 
                  for angle, intensity in fcc_data.items()}
    hcp_scaled = {angle: intensity * hcp_scaling 
                  for angle, intensity in hcp_data.items()}
    return fcc_scaled, hcp_scaled

def mixture_graph(ratio):
    """
    Returns data to plot an easily-viewable simulated spectrum
    with any provided fcc/hcp ratio"""
    fcc_scaling = ratio / (ratio + 1)
    hcp_scaling = 1 / (ratio + 1)
    fcc_scaled = fcc_scaling * fcc_intensity[window],
    hcp_scaled = hcp_scaling * hcp_intensity[window],
    return (fcc_angles[window], fcc_scaled, hcp_scaled,
            fcc_scaled + hcp_scaled)

if __name__ == '__main__':
    
    p.plot(fcc_angles[window], fcc_intensity[window])
    p.plot(hcp_angles[window], hcp_intensity[window])
    
    # Add some more info to the plot
    p.title(r'Simulated Powder XRD of FCC and HCP Hydrogen at 5498 eV')
    p.xlabel(r'$2\theta$')
    p.ylabel(r'Scattering Intensity per Cubic Angstrom')
    p.legend(['FCC','HCP'])
    p.show()
