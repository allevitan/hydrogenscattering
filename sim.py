from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as p
from lattice import *
import sys


class Sim(object):
    
    # The sim object stores the crystal structure information as
    # a class attribute - no reason to create new versions of the
    # same info for every new sim
    atomic_spacing = 3.782
    
    fcc_lattice = FCC(atomic_spacing*np.sqrt(2))#0.52)
    fcc_basis = Basis([('H',[0,0,0])])
    fcc_crystal = fcc_lattice + fcc_basis
    
    hcp_lattice = Hexagonal(atomic_spacing, atomic_spacing*np.sqrt(8/3))
    hcp_basis = Basis([('H',[0,0,0]),
                       ('H', [0.5,0.5/np.sqrt(3),np.sqrt(2/3)])],
                      l_const=atomic_spacing)
    hcp_crystal = hcp_lattice + hcp_basis


    def __init__(self, jet_profile, beam_profile, wavelength, sizes=[0.5,0.5]):
        
        # We start by storing all the raw information we've been given
        self.fcc = jet_profile[0]
        self.hcp = jet_profile[1]
        self.fcc_size = sizes[0]
        self.hcp_size = sizes[1]
        self.beam = beam_profile
        self.wavelength = wavelength

        #
        # And now we calculate and store some information about our
        # system
        #
        
        # The wavenumber, in inverse Angstrom
        self.nu = 2*np.pi/wavelength
        
        # We calculate the rlvs accessible to the radiation wavelength
        self.fcc_rlvs = find_accessible_rlvs(self.fcc_crystal,wavelength)
        self.hcp_rlvs = find_accessible_rlvs(self.hcp_crystal,wavelength)

        # We calculate the scattering angles for each rlv
        self.fcc_angles = np.round(360/np.pi*np.arcsin(np.linalg.norm(
             self.fcc_rlvs, axis=1)/(2*self.nu)),2)
        self.hcp_angles = np.round(360/np.pi*np.arcsin(np.linalg.norm(
             self.hcp_rlvs, axis=1)/(2*self.nu)),2)

        # We calculate the unit volume so we can normalize to it
        self.fcc_unit_vol = n.abs(n.dot(self.fcc_crystal.lattice[0],n.cross(
            self.fcc_crystal.lattice[1],self.fcc_crystal.lattice[2])))
        self.hcp_unit_vol = n.abs(n.dot(self.hcp_crystal.lattice[0],n.cross(
            self.hcp_crystal.lattice[1],self.hcp_crystal.lattice[2])))        
        
        # And we calculate the structure factor at each rlv, so we don't
        # need to recalculate every time it comes up
        self.fcc_s_facts = \
            n.array([n.abs(self.fcc_crystal.structure_factor(rlv)\
                           /self.fcc_unit_vol)**2
                     for rlv in self.fcc_rlvs])
        self.hcp_s_facts = \
            n.array([n.abs(self.hcp_crystal.structure_factor(rlv)\
                           /self.hcp_unit_vol)**2
                     for rlv in self.hcp_rlvs])
        
        # Remove the rlvs with s_fact=0 from HCP (FCC doesn't have any)
        good = np.logical_not(np.isclose(self.hcp_s_facts,0))
        self.hcp_rlvs = self.hcp_rlvs[good,:]
        self.hcp_s_facts = self.hcp_s_facts[good]
        
        
    def sim(self, offset, proof_plots=False):
        """
        This takes an offset between the edge of the beam profile
        and the edge of the jet profile in unts of pixels
        (0.1 um). An offset of 500 is a centred beam, an
        offset of 400 or 600 is a beam offset by 10 um. It outputs
        the spectrum captured in a simulated single shot of the beam."""
        #
        # We start by just doing geometric manipulations to turn the
        # beam and jet profiles into 2D "side-on view" arrays, and we
        # cut out all the parts of the beam profile that aren't covered
        # by the jet profile
        #
        fcc_flat = np.sum(self.fcc,axis=1)
        hcp_flat = np.sum(self.hcp,axis=1)
        beam = self.beam[offset:offset+len(fcc_flat)].transpose()
            
        #
        # Now we convert the fcc and hcp volume densities into
        # fcc and hcp volumes by multiplying by the pixel size
        # (0.1 micron)**3. We then calculate the number of
        # crystallites expected to be found in each pixel
        #
        fcc_vols = 0.1**3 * np.tile(fcc_flat,(beam.shape[0],1))
        hcp_vols = 0.1**3 * np.tile(hcp_flat,(beam.shape[0],1))
        fcc_nums = fcc_vols / self.fcc_size**3
        hcp_nums = hcp_vols / self.hcp_size**3

        if proof_plots == True:
            f, (ax1,ax2,ax3) = p.subplots(1,3)
            ax1.imshow(beam)
            ax1.set_title('Laser Fluence')
            ax2.imshow(hcp_vols)
            ax2.set_title('HCP Density')
            ax3.imshow(fcc_vols)
            ax3.set_title('FCC Density')
        
        #
        # Now we chunk the beam data into 100 bins by fluence.
        # The idea is that, in approximating the variance of the
        # spectrum, the distribution of crystallites against 
        # fluence is what really matters (1 crystallite seeing high
        # fluence is a lot more jumpy than 1000 crystallites seing
        # low fluence)
        #
        nbins = 100
        fluence_bins = np.linspace(np.min(beam[beam.nonzero()]),
                                   np.max(beam)+0.0001,nbins+1)
        fluence_slices = [(beam >= fluence_bins[i]) &
                          (beam < fluence_bins[i+1])
                          for i in range(0,nbins)]

        if proof_plots == True:
            nbins = 15
            fluence_bins = np.linspace(np.min(beam[beam.nonzero()]),
                                       np.max(beam)+0.0001,nbins+1)
            fluence_slices = [(beam >= fluence_bins[i]) &
                              (beam < fluence_bins[i+1])
                              for i in range(0,nbins)]
            fluence_showing = np.zeros(beam.shape)
            for i, fluence_slice in enumerate(fluence_slices):
                fluence_showing += i*fluence_slice
            p.figure()
            p.imshow(fluence_showing[350:-350,:])
            p.title('After Fluence Slicing (exaggerated bin size)')
        
        fcc_XRD = {}
        fcc_data = np.zeros((self.fcc_rlvs.shape[0],))
        hcp_XRD = {}
        hcp_data = np.zeros((self.hcp_rlvs.shape[0],))


        #
        # And now we actually run through, calculating the
        # spectrum from the crystallites at each fluence level.
        # Currently, the lowest fluence bin is ignored because
        # it contains too many crystallites: TODO!!!
        #
        for fluence_slice in fluence_slices[1:]:

            # Now we calculate the number of fcc and hcp crystallites
            # that will be scattering at this fluence, including some
            # randomness to fight situations where each fluence band
            # has numbers of crystals on the order of 1
            n_fcc = np.sum(fcc_nums[fluence_slice])
            n_fcc = np.floor(n_fcc) + np.int(np.random.random() < n_fcc % 1)
            n_hcp = np.sum(hcp_nums[fluence_slice])
            n_hcp = np.floor(n_hcp) + np.int(np.random.random() < n_hcp % 1)
            
            # We calculate the average fluence seen by the fcc and
            # hcp crystallites in this fluence slice
            if n_fcc != 0:
                fcc_avg = np.sum(fcc_nums[fluence_slice] *
                                 beam[fluence_slice]) / n_fcc
            if n_hcp != 0:
                hcp_avg = np.sum(hcp_nums[fluence_slice] * 
                                 beam[fluence_slice]) / n_hcp
                        
            # And now we actually simulate the diffraction!
            if n_fcc != 0:
                fcc_data += hardcore_powder_XRD(
                    self.fcc_crystal, self.wavelength,
                    n_fcc,self.fcc_size,
                    rlvs=self.fcc_rlvs, s_facts=self.fcc_s_facts) * fcc_avg
            if n_hcp != 0:    
                hcp_data += hardcore_powder_XRD(
                    self.hcp_crystal, self.wavelength,
                    n_hcp,self.hcp_size,
                    rlvs=self.hcp_rlvs, s_facts=self.hcp_s_facts) * hcp_avg

        #
        # Now we enter the part of the code where we package up
        # the scattering data into nice, manageable dictionaries.
        # this is also where we collapse different RLVS at the same
        # scattering angle
        #
        for angle, intensity in zip(self.fcc_angles,fcc_data):
            #if np.isclose(intensity,0):
            #    continue
            try:
                fcc_XRD[angle] += intensity 
            except KeyError:
                fcc_XRD[angle] = intensity 
 
        for angle, intensity in zip(self.hcp_angles,hcp_data):
            #if np.isclose(intensity,0):
            #    continue
            try:
                hcp_XRD[angle] += intensity 
            except KeyError:
                hcp_XRD[angle] = intensity 
        
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
    fcc_data, hcp_data = sim.sim(600)
    max_fcc = max(fcc_data.values())
    max_hcp = max(hcp_data.values())
    print('\n\nFCC\n---')
    for angle, intensity in sorted(fcc_data.items()):
        print(str(angle) + ':', intensity/max_fcc)
    print('\n\nHCP\n---')
    for angle, intensity in sorted(hcp_data.items()):
        print(str(angle) + ':', intensity/max_hcp)
    fcc_angles, fcc_intensity = spectrumify(fcc_data)
    hcp_angles, hcp_intensity = spectrumify(hcp_data)
    p.plot(fcc_angles, fcc_intensity)
    p.plot(hcp_angles, hcp_intensity)
    p.show()

