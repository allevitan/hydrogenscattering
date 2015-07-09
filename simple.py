from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from sim import Sim


xs = np.linspace(-5,5,101)
ys = np.linspace(-5,5,101)
Xs,Ys = np.meshgrid(xs,ys)
fcc = np.logical_and((np.sqrt(Xs**2 + Ys**2) <= 2.5),
                     (np.sqrt(Xs**2 + Ys**2) > 2)).astype(float)
hcp = (np.sqrt(Xs**2 + Ys**2) <2).astype(float)


beam_xs = np.linspace(-75,75,1101)
beam_zs = np.linspace(-75,75,1101)
beam_Xs, beam_Zs = np.meshgrid(beam_xs,beam_zs)
beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(20/2.355)**2))
wide_beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(50/2.355)**2))
double_beam = 0.9*beam + 0.09*wide_beam + 0.01


sim = Sim((fcc,hcp),double_beam)


fccs,hcps = [],[]
offsets = np.arange(-500,500)
for offset in offsets:
    fcc, hcp = sim.sim(500-offset)
    fccs.append(np.sum(fcc))
    hcps.append(np.sum(hcp))

fccs = np.array(fccs)
hcps = np.array(hcps)


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

p.figure()
p.plot(0.1*offsets, fccs/hcps)
p.title('Relative contributions (FCC/HCP) with 0.5um FCC sheath')
p.xlabel('FEL offset from jet center (um)')
p.ylabel('Ratio of FCC to HCP contribution')



new_Xs, new_Zs = np.meshgrid(xs,beam_zs)

fcc25, hcp25 = sim.sim(500-250)

p.figure()
p.pcolormesh(new_Xs,new_Zs,fcc25)
p.colorbar()
p.title('FCC contribution, FEL offset by 25 um')

p.figure()
p.pcolormesh(new_Xs,new_Zs,hcp25)
p.colorbar()
p.title('HCP contribution, FEL offset by 25 um')

fcc10, hcp10 = sim.sim(500-100)

p.figure()
p.pcolormesh(new_Xs,new_Zs,fcc10)
p.colorbar()
p.title('FCC contribution, FEL offset by 10 um')

p.figure()
p.pcolormesh(new_Xs,new_Zs,hcp10)
p.colorbar()
p.title('HCP contribution, FEL offset by 10 um')
p.show()


