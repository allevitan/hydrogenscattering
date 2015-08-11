from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as p
from sim import *

xs = np.linspace(-5,5,101)
ys = np.linspace(-5,5,101)
Xs,Ys = np.meshgrid(xs,ys)
fcc_jet = (np.sqrt(Xs**2 + Ys**2) <= 2.5).astype(float)*0.4
hcp_jet = (np.sqrt(Xs**2 + Ys**2) <= 2.5).astype(float)*0.6



beam_xs = np.linspace(-75,75,1101)
beam_zs = np.linspace(-75,75,1101)
beam_Xs, beam_Zs = np.meshgrid(beam_xs,beam_zs)
beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(10/2.355)**2))
wide_beam = np.exp(-(beam_Xs**2 + beam_Zs**2)/(2*(30/2.355)**2))
double_beam = 0.9*beam + 0.099*wide_beam + 0.001


sim = Sim((fcc_jet,hcp_jet),double_beam, 2.265, [0.44,0.53])




max_ints = []
all_hcp_angles = []
all_hcp_ints = []
all_fcc_angles = []
all_fcc_ints = []
for i in range(500):
    ((fcc_angles,fcc_offsets,fcc_s_facts),
     (hcp_angles, hcp_offsets, hcp_s_facts)) \
        = sim.sim(500, detector_angle=0.5, direct=True)
    d = n.pi / (0.1*10000)
    nu = 2*n.pi/sim.wavelength
    graph_angles = n.linspace(0,n.pi,10000)
    hcp_sum_ints = []
    for angle, offset, s_fact in zip(hcp_angles, hcp_offsets, hcp_s_facts):
        int_squared = n.maximum(0,d**2 - offset**2 - 
                                ((angle-graph_angles)*nu)**2)
        graph_ints = n.sqrt(int_squared) * s_fact
        hcp_sum_ints.append(sum(graph_ints))
    fcc_sum_ints = []
    for angle, offset, s_fact in zip(fcc_angles, fcc_offsets, fcc_s_facts):
        int_squared = n.maximum(0,d**2 - offset**2 - 
                                ((angle-graph_angles)*nu)**2)
        graph_ints = n.sqrt(int_squared) * s_fact
        fcc_sum_ints.append(sum(graph_ints))

    all_hcp_angles.append(hcp_angles)
    all_hcp_ints.append(hcp_sum_ints)
    all_fcc_angles.append(fcc_angles)
    all_fcc_ints.append(fcc_sum_ints)
    max_ints.append(max(hcp_sum_ints + [0]))
    max_ints.append(max(fcc_sum_ints + [0]))
    

    print(i)

max_int = max(max_ints)
p1hits = []
p2hits = []
for angles, sum_ints in zip(all_hcp_angles,all_hcp_ints):
    p1hits.append(0)
    p2hits.append(0)
    for angle, sum_int in zip(angles, sum_ints):
        if max_int/sum_int < 1e3 and abs(angle*180/n.pi - 40.5) < 1:
            p1hits[-1] += 1
        if max_int/sum_int < 1e3 and abs(angle*180/n.pi - 43) < 1:
            p2hits[-1] += 1
for i, (angles, sum_ints) in enumerate(zip(all_fcc_angles,all_fcc_ints)):
    for angle, sum_int in zip(angles, sum_ints):
        if max_int/sum_int < 1e3 and abs(angle*180/n.pi - 40.5) < 1:
            p1hits[i] += 1
        if max_int/sum_int < 1e3 and abs(angle*180/n.pi - 43) < 1:
            p2hits[i] += 1



print(p1hits)
print(p2hits)
p.hist(p1hits,range=[-0.5,20.5],bins=21)
p.figure()
p.hist(p2hits,range=[-0.5,20.5],bins=21)
p.show()
