from __future__ import print_function
import numpy as np
from lattice import *
from time import time
import sys


class Sim(object):
    atomic_spacing = 3.78
    
    fcc_lattice = FCC(atomic_spacing*np.sqrt(2))#0.52)
    fcc_basis = Basis([('H',[0,0,0])])
    fcc_crystal = fcc_lattice + fcc_basis
    
    
    hcp_lattice = Hexagonal(atomic_spacing, atomic_spacing*np.sqrt(8/3))
    hcp_basis = Basis([('H',[0,0,0]),
                       ('H', [0.5,0.5/np.sqrt(3),np.sqrt(2/3)])],
                      l_const=atomic_spacing)
    hcp_crystal = hcp_lattice + hcp_basis


    def __init__(self, jet_profile, beam_profile, wavelength, sizes=[.1,.1]):
        self.fcc = jet_profile[0]
        self.hcp = jet_profile[1]
        self.fcc_size = sizes[0]
        self.hcp_size = sizes[1]
        self.beam = beam_profile
        self.wavelength = wavelength
        self.nu = 2*np.pi/wavelength
        
        self.fcc_rlvs = find_accessible_rlvs(self.fcc_crystal,wavelength)
        self.hcp_rlvs = find_accessible_rlvs(self.hcp_crystal,wavelength)
        self.fcc_unit_vol = n.abs(n.dot(self.fcc_crystal.lattice[0],n.cross(
            self.fcc_crystal.lattice[1],self.fcc_crystal.lattice[2])))
        self.hcp_unit_vol = n.abs(n.dot(self.hcp_crystal.lattice[0],n.cross(
            self.hcp_crystal.lattice[1],self.hcp_crystal.lattice[2])))        
        self.fcc_s_facts = n.array([n.abs(self.fcc_crystal.structure_factor(rlv)\
                                  /self.fcc_unit_vol)**2
                            for rlv in self.fcc_rlvs])
        self.hcp_s_facts = n.array([n.abs(self.hcp_crystal.structure_factor(rlv)\
                                  /self.hcp_unit_vol)**2
                                    for rlv in self.hcp_rlvs])

        
        
        
    def sim(self, offset):
        fcc_flat = np.sum(self.fcc,axis=1)
        hcp_flat = np.sum(self.hcp,axis=1)
        beam = self.beam[offset:offset+len(fcc_flat)].transpose()
        fcc_vols = 0.1**3 * np.tile(fcc_flat,(beam.shape[0],1))
        hcp_vols = 0.1**3 * np.tile(hcp_flat,(beam.shape[0],1))
        fcc_nums = np.round(fcc_vols / self.fcc_size**3)
        hcp_nums = np.round(hcp_vols / self.hcp_size**3)
        fcc_worthit = np.nonzero(fcc_nums)
        hcp_worthit = np.nonzero(hcp_nums)
        fcc_angles = np.round(360/np.pi*np.arcsin(np.linalg.norm(
            self.fcc_rlvs, axis=1)/(2*self.nu)),2)

        fcc_XRD = {}
        fcc_data = np.zeros((self.fcc_rlvs.shape[0],))
        n = 0
        for fcc_num, fluence in zip(fcc_nums[fcc_worthit].flatten(),
                                    beam[fcc_worthit].flatten()):
            if n % 1000 == 0:
                sys.stdout.write('\rCalculated ' + str(n) + 
                                 '/' + str(len(fcc_worthit[0])) + ' of FCC')
                sys.stdout.flush()
            n += 1
            t0 = time()
            fcc_data += hardcore_powder_XRD(
                self.fcc_crystal, self.wavelength,
                fcc_num,self.fcc_size,
                rlvs=self.fcc_rlvs, s_facts=self.fcc_s_facts) * fluence
        
        for angle, intensity in zip(fcc_angles,fcc_data):
            if np.isclose(intensity,0):
                continue
            try:
                fcc_XRD[angle] += intensity 
            except KeyError:
                fcc_XRD[angle] = intensity 
        sys.stdout.write('\rFinished with FCC              \n')
        sys.stdout.flush()

        hcp_XRD = {}
        hcp_data = np.zeros((self.hcp_rlvs.shape[0],))
        n = 0
        for hcp_num, fluence in zip(hcp_nums[hcp_worthit].flatten(),
                                    beam[hcp_worthit].flatten()):
            if n % 1000 == 0:
                sys.stdout.write('\rCalculated ' + str(n) + 
                                 '/' + str(len(hcp_worthit[0])) + ' of HCP')
                sys.stdout.flush()
            n += 1
            t0 = time()
            hcp_data += hardcore_powder_XRD(
                self.hcp_crystal, self.wavelength,
                hcp_num,self.hcp_size,
                rlvs=self.hcp_rlvs, s_facts=self.hcp_s_facts) * fluence
        
        hcp_angles = np.round(360/np.pi*np.arcsin(np.linalg.norm(
            self.hcp_rlvs, axis=1)/(2*self.nu)),2)

        for angle, intensity in zip(hcp_angles,hcp_data):
            if np.isclose(intensity,0):
                continue
            try:
                hcp_XRD[angle] += intensity 
            except KeyError:
                hcp_XRD[angle] = intensity 
        sys.stdout.write('\rFinished with HCP             \n')
        sys.stdout.flush()
        
        return fcc_XRD, hcp_XRD
            


if __name__ == '__main__':
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
    
    # And we set up the "simulation"
    sim = Sim((fcc_jet,hcp_jet),double_beam,2.255)
    print('Set up sim')
    fcc_data, hcp_data = sim.sim(500)
    max_fcc = max(fcc_data.values())
    max_hcp = max(hcp_data.values())
    print('\n\nFCC\n---')
    for angle, intensity in sorted(fcc_data.items()):
        print(str(angle) + ':', intensity/max_fcc)
    print('\n\nHCP\n---')
    for angle, intensity in sorted(hcp_data.items()):
        print(str(angle) + ':', intensity/max_hcp)
