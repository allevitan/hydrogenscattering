from __future__ import print_function, division
from matplotlib import pyplot as p
from matplotlib import rc
from lattice import *

# Set up the crystal structure
lattice = FCC(3.5597)
basis = Basis([('C',[0,0,0]),('C',[0.25,0.25,0.25])],l_const=3.5597)
crystal = lattice + basis

atomic_spacing = 3.78
#lattice = Hexagonal(atomic_spacing, atomic_spacing*n.sqrt(8/3))
lattice = Hexagonal(2.659,4.334)
basis = Basis([('H',[0,0,0]),
               ('H', [0.5,0.5/n.sqrt(3),n.sqrt(2/3)])],
                  l_const=2.659)
crystal = lattice + basis

# Plot a simulated XRD with copper radiation
scattering_data = hardcore_powder_XRD(crystal,1.5405,200000,1000, niceify=True)
#scattering_data = hardcore_powder_XRD(crystal,2.255,20000,1000, niceify=True)
scattering_data = scattering_data
angles, values = spectrumify(scattering_data)
p.plot(angles,values)

norm_fact = max(scattering_data.values())
for angle, intensity in sorted(scattering_data.items()):
    if n.isclose(intensity,0):
        continue
    print('%.2f' % angle + ':', intensity/norm_fact*100)

# Add some more info to the plot
p.title(r'Simulated Powder XRD of Diamond, $\lambda = 1.5405$')
p.xlabel(r'$2\theta$')
p.ylabel(r'Scattering Intensity per Cubic Angstrom')
p.show()
