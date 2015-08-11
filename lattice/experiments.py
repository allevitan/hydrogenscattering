from __future__ import division, print_function
import numpy as n
import itertools as it
from time import time


def find_accessible_rlvs(crystal, wavelength):
    # We generate a list of accessible reciprocal lattice
    # vectors. To be accessible, the magnitude of a rlv's
    # wavevector must be less than twice that of the input
    # radiation's wavenumber.
    
    #The input wavenumber.
    nu = 2*n.pi/wavelength 
    # Now we find the shortest distance to a wall of a 
    # parallelogram "shell" in the reciprocal lattice
    min_step = min(abs(n.dot(
        (crystal.rlattice[0]+crystal.rlattice[1]
         +crystal.rlattice[2]),
        n.cross(crystal.rlattice[i],crystal.rlattice[j])
        /n.linalg.norm(n.cross(crystal.rlattice[i],crystal.rlattice[j]))))
                   for i,j in [(0,1),(1,2),(2,0)])
    # If we look at all the points in this many parallelogram
    # "shells", we can't miss all the accessible wavevectors
    num_shells = int(2*nu / min_step)
    # Now we generate these possibilities
    possibilities = [(crystal.rlattice[0]*h + crystal.rlattice[1]*j
                      + crystal.rlattice[2]*k)
                     for h,j,k in it.product(
                             range(-num_shells,num_shells+1),
                             repeat=3)]
    # And we filter the possibilities, getting rid of all the
    # rlvs that are too long and the 0 vector
    rlvs = [rlv for rlv in possibilities 
            if n.linalg.norm(rlv) < 2*nu
            and not n.allclose(rlv,0)]
    
    return n.array(rlvs)


def powder_XRD(crystal,wavelength, get_mults=False):
    """
    Generates a powder XRD spectrum for radiation with the
    given wavelength (in angstroms)
    """
    nu = 2*n.pi/wavelength
    rlvs = find_accessible_rlvs(crystal,wavelength)

    # Now we renormalize the intensities to account for the fact that
    # the same lattice can be described by different unit cells
    unit_vol = n.abs(n.dot(crystal.lattice[0],n.cross(
        crystal.lattice[1],crystal.lattice[2])))
    
    # Now we calculate the scattering intensity from each rlv
    intensities = {
        tuple(rlv): n.abs(crystal.structure_factor(rlv)/unit_vol)**2
        for rlv in rlvs}
    
    # We actually only care about the magnitudes of the rlvs
    magnitudes = {}
    multiplicities = {}
    for rlv, intensity in intensities.items():
        repeat = False
        mag = n.linalg.norm(rlv)
        for oldmag in magnitudes:
            if n.isclose(mag,oldmag):
                magnitudes[oldmag] += intensity
                multiplicities[oldmag] += 1
                repeat = True
                break
        if not repeat:
            multiplicities[mag] = 1
            magnitudes[mag] = intensity
        
    # Now we calculate the scattering angles and intensities
    multiplicities = {2 * n.arcsin(mag / (2 * nu)) * 180 / n.pi:
                      multiplicity
                      for mag, multiplicity in multiplicities.items()
                      if not n.allclose(magnitudes[mag],0)}
    intensities = {2 * n.arcsin(mag / (2 * nu)) * 180 / n.pi:
                   intensity * 
                   # This factor corrects for the fact that the same total
                   # power in the debye scherrer rings is more
                   # concentrated at 2\theta =  0 and 2pi
                   1 / n.sin(2*n.arcsin(mag/(2*nu))) *
                   # This factor corrects for the angular dependence of
                   # scattering probability given an equal incident
                   # scattering wavevector and an equal alowed variance
                   # around the scattering vector
                   1 / mag * #cos(theta)/(cos(theta)*sin(theta))
                   # This factor corrects for polarization effects,
                   # Assuming an unpolarized input beam and no polarization
                   # analysis
                   (1 + n.cos(2*n.arcsin(mag/(2*nu)))**2)/2
                   for mag, intensity in magnitudes.items()
                   if not n.allclose(intensity,0)}
    if get_mults:
        return intensities, multiplicities
    else:
        return intensities

def hardcore_powder_XRD(crystal, wavelength, num, l, rlvs=None, s_facts=None, niceify=False, direct=False, detector_angle=None):

    
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



def spectrumify(scattering_data):
    """
    This is just a nice function to turn the raw scattering data
    into a human-readable approximation of a scattering spectrum
    """
    graph_angles = n.linspace(0,180,10000)
    graph_intensities = n.zeros(graph_angles.shape)
    
    for angle, intensity in sorted(scattering_data.items()):
        graph_intensities += intensity * \
                             n.exp(-(graph_angles - angle)**2 / (2*(0.1)**2))
        
    return graph_angles, graph_intensities




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
