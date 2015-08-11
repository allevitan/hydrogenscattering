from __future__ import division, print_function
import numpy as n
from lattice import *
import itertools as it
from time import time


def monte_carlo_XRD(crystal, wavelength, num, l, rlvs=None, s_facts=None, niceify=False, direct=False, detector_angle=None):

    
    d = n.pi/n.float(l*10000) #10000 is a conversion from um to Angstrom
    nu = 2*n.pi/wavelength

    # Now we renormalize the intensities to account for the fact that
    # the same lattice can be described by different unit cells
    unit_vol = n.abs(n.dot(crystal.lattice[0],n.cross(
        crystal.lattice[1],crystal.lattice[2])))

    # If pre-calculated rlvs and structure factors aren't provided,
    # calculate them now
    if rlvs == None:
        rlvs = find_accessible_rlvs(crystal,wavelength)
    if s_facts == None:
        s_facts = n.array([n.abs(crystal.structure_factor(rlv)/unit_vol)**2
                   for rlv in rlvs])
    
    # Now we generate <num> random vectors with length k (equivalent to
    # generating <num> randomly oriented sets of rlvs for the same k)
    ks = n.random.rand(3,num) - 0.5
    #phi_0s = n.random.rand(num) * 2*n.pi
    ks = (ks / n.linalg.norm(ks, axis=0) * nu).transpose()
    
    # Now we calculate the scattering vector that would be needed to
    # acccess each rlv
    kprimes = rlvs[:,None,:] - ks
    #ang_vecs = n.cross(kprimes,ks[None,:,:])
    #zero_angs = n.cross([1,0,0],ks)
    #ang_vecs = ang_vecs / n.linalg.norm(ang_vecs,axis=2)[:,:,None]
    #zero_angs = zero_angs / n.linalg.norm(zero_angs,axis=1)[:,None]
    #dots = n.sum(ang_vecs*zero_angs[None,:,:],axis=2)
    #crosses = n.sum(ks[None,:,:]*n.cross(ang_vecs,zero_angs[None,:,:]),axis=2)/nu
    #az_angs =  n.pi*(1 - n.sign(dots))/2 + n.arcsin(crosses)*n.sign(dots) - phi_0s
    
    
    # And we calculate the difference in wavevector between that scattering
    # vector and what it needs to be
    knorms = n.linalg.norm(kprimes,axis=2)
    offsets = knorms - nu
    
    if direct:
        picks = n.abs(offsets) < d
        #detector_angle bit STILL NEEDS TO BE TESTED
        if detector_angle is not None:
            picks = picks & (n.random.rand(*picks.shape)<detector_angle/(2*n.pi))
        calcangles = n.arccos(n.sum(-kprimes*ks,axis=2) / (nu*knorms))
        picked_angles = calcangles[picks]
        picked_offsets = offsets[picks]
        picked_s_facts = (s_facts.transpose() *
                          n.ones(calcangles.shape).transpose()).transpose()[picks]
        return picked_angles, picked_offsets, picked_s_facts

    
    intensities = l**2 * (((d + offsets) / (4*(nu+offsets))).transpose() \
                  * s_facts.transpose()).transpose() * l**3
    intensities[n.abs(offsets)>d] = 0
    # Just assumes each vector has an equal chance of scattering into the
    # availible angle
    if detector_angle is not None:
        intensities[n.random.rand(*intensities.shape)>detector_angle/(2*n.pi)] = 0

    angles = n.arcsin(n.linalg.norm(rlvs, axis=1)/(2*nu))
    #calcangs = n.average(calcangles,weights=offsets<d)
    intensities = n.sum(intensities, axis=1)
    intensities = intensities / n.sin(2*angles) * \
                  (1 + n.cos(2*angles)**2)
    
    if niceify:
        final_data = {}
        for angle, intensity in zip(angles, intensities):
            # These factors account for the size of the debye-scherrer ring
            # and the polarization effects (assuming no polarization analysis)
            old_angles = final_data.keys()
            repeats = n.isclose(old_angles,round(360/n.pi*angle,2))
            if any(repeats):
                final_data[old_angles[n.nonzero(repeats)[0][0]]] += \
                                                    intensity
            else:
                final_data[round(360/n.pi*angle,2)] = intensity
        return final_data
    else:
        return intensities


def gen_full_spectrum(angles, offsets, s_facts, crystal_size, wavelength):
    # phis = n.linspace(0,n.pi/2,1000)
    # angle_grid = n.linspace(0,n.pi,500)
    # offset_grid = n.linspace(-3*n.pi/crystal_size,3*n.pi/crystal_size,500)
    # angle_grid,offset_grid = n.meshgrid(angle_grid,offset_grid)
    
    
    
    d = n.pi / (crystal_size*10000)
    nu = 2*n.pi/wavelength
    maxoff = n.amax(offsets)
    graph_angles = n.linspace(0,n.pi,10000)
    graph_intensities = n.zeros(graph_angles.shape)
    for angle, offset, s_fact in zip(angles, offsets, s_facts):
        int_squared = n.maximum(0,d**2 - offset**2 - 
                                ((angle-graph_angles)*nu)**2)
        graph_intensities += n.sqrt(int_squared) * s_fact
    return graph_angles*180/n.pi, graph_intensities
