from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
from sim import *
from XRD_math import *

#
# We start by setting up the hydrogen jet. It has 2 parts:
# an FCC intensity and an HCP intensity part
#

xs = np.linspace(-5,5,101)
ys = np.linspace(-5,5,101)
Xs,Ys = np.meshgrid(xs,ys)
hcp_jet = (np.sqrt(Xs**2 + Ys**2) <2.5).astype(float)
fcc_jet = np.zeros(hcp_jet.shape).astype(float)
#fcc_jet = np.logical_and((np.sqrt(Xs**2 + Ys**2) <= 2.5),
#                         (np.sqrt(Xs**2 + Ys**2) >= 1.75)).astype(float)
#hcp_jet = (np.sqrt(Xs**2 + Ys**2) <1.74).astype(float)

#
# Now we set up the beam profile, as a distribution of fluence
#

beam_xs = np.linspace(-75,75,1101)
beam_zs = np.linspace(-75,75,1101)
beam_Xs, beam_Zs = np.meshgrid(beam_xs,beam_zs)
beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(10/2.355)**2))
wide_beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(30/2.355)**2))
double_beam = 0.9*beam + 0.01*wide_beam #+ 0.001


# And we set up the "simulation"
sim = Sim((fcc_jet,hcp_jet),double_beam,2.255,[1,0.3])
fcc_data = []
hcp_data = []
for i in range(0,500):
    print(i)
    fcc_datum, hcp_datum = sim.sim(500)
    fcc_data.append(fcc_datum)
    hcp_data.append(hcp_datum)

hcp_percents = []
for fcc_datum, hcp_datum in zip(fcc_data, hcp_data):
    peak_1 = sorted(hcp_datum.items())[1][1]
    peak_2 = sorted(fcc_datum.items())[0][1] + \
             sorted(hcp_datum.items())[2][1]
    hcp_percents.append(calc_HCP_percentage(peak_1,peak_2))

np.savetxt('0.5um_fcc_big.csv',np.array(hcp_percents))


print(hcp_percents)
    

