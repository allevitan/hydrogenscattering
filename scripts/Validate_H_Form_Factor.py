from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from lattice import *


#
#
# This script is testing whether the scattering form factor
# for atomic hydrogen can well approximate the scattering form
# factor for molecular hydrogen (2*H \approx H2?).
#
# The crystal structure and probe radiation are the same as were
# used in Mao H K, Jephcoat A P, Hemley R J, Finger L W, 
# Zha C S, Hazen R M, Cox D E Science 239 (1988) 1131-1133
#
# The resulting peak intensities are to be compared to the 
# following normalized scattering data from that paper:
# 
#   2-THETA      INTENSITY    D-SPACING   H   K   L   Multiplicity
#    39.12         33.11        2.3028    1   0   0         6
#    41.68         31.85        2.1670    0   0   2         2
#    44.56        100.00        2.0335    1   0   1        12
#    58.48          6.49        1.5781    1   0   2        12
#    70.88          3.62        1.3295    1   1   0         6
#    78.09          2.84        1.2238    1   0   3        12
#    85.73          2.09        1.1332    1   1   2        12
#    87.70          1.37        1.1128    2   0   1        12
#
#

# Set up the crystal structure
atomic_spacing = 2.659

BS = 0.00

a = atomic_spacing*np.array([1,0,0])
b = atomic_spacing*np.array([-0.5,np.sqrt(3)/2,0])
c = atomic_spacing*np.array([0,0,np.sqrt(8/3)])
hcp_lattice = Lattice(a,b,c)
hcp_basis = Basis([('H',[0,0,0]),
                   ('H',[0.5,0.5/np.sqrt(3),np.sqrt(2/3)])],
                   l_const=atomic_spacing)
hcp_crystal = hcp_lattice + hcp_basis


hcp_data = powder_XRD(hcp_crystal,1.5418)
hcp_angles, hcp_intensity = spectrumify(hcp_data)


for angle, intensity in sorted(hcp_data.items()):
    if angle <= 90:
        print("%.2f" % angle + ':', 
              "%.2f" % (intensity / max(hcp_data.values()) * 100))
