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

sim = Sim((fcc,hcp),beam)


fccs,hcps = [],[]
offsets = np.arange(-500,500)
for offset in offsets:
    fcc, hcp = sim.sim(500-offset)
    fccs.append(np.sum(fcc))
    hcps.append(np.sum(hcp))

fccs = np.array(fccs)
hcps = np.array(hcps)

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
p.show()


# fcc, hcp = sim.sim(200)

# fcctotal = np.sum(fcc)
# hcptotal = np.sum(hcp)
# print('fcc',fcctotal)
# print('hcp',hcptotal)

# new_Xs, new_Zs = np.meshgrid(xs,beam_zs)
# p.pcolormesh(new_Xs,new_Zs,fcc)
# p.colorbar()
# p.figure()
# p.pcolormesh(new_Xs,new_Zs,hcp)
# p.colorbar()
# p.show()
