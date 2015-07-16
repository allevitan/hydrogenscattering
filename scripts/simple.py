from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from sim import Sim


#
# We start by setting up the hydrogen jet. It has 2 parts:
# an FCC intensity and an HCP intensity part
#

xs = np.linspace(-5,5,101)
ys = np.linspace(-5,5,101)
Xs,Ys = np.meshgrid(xs,ys)
fcc_jet = np.logical_and((np.sqrt(Xs**2 + Ys**2) <= 2.5),
                         (np.sqrt(Xs**2 + Ys**2) >= 2)).astype(float)
hcp_jet = (np.sqrt(Xs**2 + Ys**2) <2).astype(float)


#
# Now we set up the beam profile, as a distribution of fluence
#

beam_xs = np.linspace(-75,75,1101)
beam_zs = np.linspace(-75,75,1101)
beam_Xs, beam_Zs = np.meshgrid(beam_xs,beam_zs)
beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(10/2.355)**2))
wide_beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(30/2.355)**2))
double_beam = 0.9*beam + 0.099*wide_beam + 0.001

p.figure()
p.pcolor(Xs,Ys,fcc_jet + 0.5*hcp_jet)
p.xlabel('X distance (um)')
p.ylabel('Y distance (um)')
p.title('Generic H2 Jet Structure')


# And we set up the "simulation"
sim = Sim((fcc_jet,hcp_jet),double_beam)

#
# Now we pull out key parameters for simulations involving a
# wide range of offsets
#

fccs,hcps = [],[]
offsets = np.arange(-300,300)
for offset in offsets:
    fcc, hcp = sim.sim(500-offset)
    fccs.append(np.sum(fcc))
    hcps.append(np.sum(hcp))

fccs = np.array(fccs)
hcps = np.array(hcps)

p.figure()
p.plot(beam_xs,double_beam[:,550],'b-')
p.plot(beam_xs,0.9*beam[:,550],'g--')
p.plot(beam_xs,0.1*wide_beam[:,550],'r--')
p.plot(beam_xs,0*beam_xs+0.01,'k--')
p.legend(['sum','narrow (FWHM=20)','wide (FWHM=50)','background'])
p.title('Generic beam profile')
p.ylabel('Fluence (arbitrary units)')
p.xlabel('Radial distance (um)')

p.figure()
p.plot(0.1*offsets, fccs)
p.plot(0.1*offsets, hcps)
p.title('FCC and HCP contributions with 0.5um FCC sheath')
p.xlabel('FEL offset from jet center (um)')
p.ylabel('Contribution to scattering (arbitrary units)')
p.legend(['FCC','HCP'])


crude_approx = 0*offsets+(np.sum(fcc_jet)/np.sum(hcp_jet) \
               + np.diff(double_beam[550,:],n=2)[50:-49]*20 \
               / double_beam[550,51:-50])[500+offsets]
p.figure()
p.plot(0.1*offsets, fccs/hcps,'b-')
p.plot(0.1*offsets,0*offsets+np.sum(fcc_jet)/np.sum(hcp_jet), 'g--')
p.plot(0.1*offsets, crude_approx, 'r--')
p.title('Relative contributions (FCC/HCP) with 0.5um FCC sheath')
p.xlabel('FEL offset from jet center (um)')
p.ylabel('Ratio of FCC to HCP contribution')
p.legend(['Simulated ratio','Volume ratio','Crude approximation'],loc=4) 


#
# And now we pull some more detailed information from 2
# simulations with particular offests
#


new_Xs, new_Zs = np.meshgrid(xs,beam_zs)

fcc15, hcp15 = sim.sim(500-150)

p.figure()
p.pcolormesh(new_Xs,new_Zs,fcc15)
p.colorbar()
p.title('FCC contribution, FEL offset by 15 um')

p.figure()
p.pcolormesh(new_Xs,new_Zs,hcp15)
p.colorbar()
p.title('HCP contribution, FEL offset by 15 um')

fcc7, hcp7 = sim.sim(500-70)

p.figure()
p.pcolormesh(new_Xs,new_Zs,fcc7)
p.colorbar()
p.title('FCC contribution, FEL offset by 7 um')

p.figure()
p.pcolormesh(new_Xs,new_Zs,hcp7)
p.colorbar()
p.title('HCP contribution, FEL offset by 7 um')
p.show()

