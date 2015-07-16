from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *


# Set up the crystal structure
atomic_spacing = 2.659

BS = 0.00

a = atomic_spacing*np.array([1,0,0])
b = atomic_spacing*np.array([-0.5,np.sqrt(3)/2,0])
c = atomic_spacing*np.array([0,0,np.sqrt(8/3)])
hcp_lattice = Lattice(a,b,c)
#hcp_basis = Basis([('H',[0,0,BS/atomic_spacing]),
#                   ('H',[0,0,-BS/atomic_spacing]),
#                   ('H',[0.5,0.5/np.sqrt(3),np.sqrt(2/3)+BS/atomic_spacing]),
#                   ('H',[0.5,0.5/np.sqrt(3),np.sqrt(2/3)-BS/atomic_spacing])],
#                   l_const=atomic_spacing)
hcp_basis = Basis([('H2',[0,0,0]),
                   ('H2',[0.5,0.5/np.sqrt(3),np.sqrt(2/3)])],
                   l_const=atomic_spacing)
hcp_crystal = hcp_lattice + hcp_basis


hcp_data = powder_XRD(hcp_crystal,1.5418)
hcp_angles, hcp_intensity = spectrumify(hcp_data)


for angle, intensity in sorted(hcp_data.items()):
    if angle <= 90:
        print("%.2f" % angle + ':', 
              "%.2f" % (intensity / max(hcp_data.values()) * 100))
