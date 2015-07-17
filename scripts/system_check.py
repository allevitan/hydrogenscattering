from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from sim import *
from XRD_math import *


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

p.figure(figsize=(8,8))
p.pcolormesh(Xs,Ys,fcc_jet + 0.5*hcp_jet)
p.xlim([np.min(Xs),np.max(Xs)])
p.ylim([np.min(Ys),np.max(Ys)])
p.xlabel('X distance (um)')
p.ylabel('Y distance (um)')
p.title('Generic H2 Jet Structure')

p.figure(figsize=(8,8))
p.imshow(double_beam,extent=(np.min(beam_Xs),np.max(beam_Xs),
                             np.min(beam_Zs),np.max(beam_Zs)))
p.xlabel('X distance (um)')
p.ylabel('Z distance (um)')
p.title('Generic Laser Spot')

p.figure()
p.plot(beam_xs,double_beam[:,550],'b-')
p.plot(beam_xs,0.9*beam[:,550],'g--')
p.plot(beam_xs,0.099*wide_beam[:,550],'r--')
p.plot(beam_xs,0*beam_xs+0.001,'k--')
p.legend(['sum','narrow (FWHM=10)','wide (FWHM=30)','background'])
p.title('Generic beam profile')
p.ylabel('Fluence (arbitrary units)')
p.xlabel('Radial distance (um)')


# And we set up the "simulation"
sim = Sim((fcc_jet,hcp_jet),double_beam, 2.265, [0.5,0.5])

fcc_data, hcp_data = sim.sim(450,proof_plots=True)

angles_ideal, fcc_ideal, hcp_ideal, sum_ideal = mixture_graph(1)
hcp_norm_ideal = hcp_ideal / np.max(hcp_ideal)
fcc_norm_ideal = fcc_ideal / np.max(fcc_ideal)
p.figure()
p.plot(angles_ideal,hcp_ideal,'b-')
p.plot(angles_ideal,fcc_ideal,'r-')
p.xlabel('Scattering Angle $2\theta$')
p.ylabel('Intensity (arbitrary units)')
p.legend(['HCP Contribution','FCC Contribution'])
p.title('Simulated Ideal Spectrum of 1:1 HCP:FCC Mixture')


n_cryst = 100000

hcp_sim = hardcore_powder_XRD(sim.hcp_crystal,sim.wavelength,n_cryst,0.5,
                              niceify=True)
fcc_sim = hardcore_powder_XRD(sim.fcc_crystal,sim.wavelength,n_cryst,0.5,
                              niceify=True)
hcp_sim_angles, hcp_sim_intensities = spectrumify(hcp_sim)
fcc_sim_angles, fcc_sim_intensities = spectrumify(fcc_sim)
hcp_sim_norm_ints = hcp_sim_intensities / np.max(hcp_sim_intensities)
fcc_sim_norm_ints = fcc_sim_intensities / np.max(fcc_sim_intensities)
window = (hcp_sim_angles > 35) & (hcp_sim_angles < 55)
p.figure()
p.plot(hcp_sim_angles[window], hcp_sim_norm_ints[window],'b')
p.plot(angles_ideal, hcp_norm_ideal, 'k--')
p.xlabel('Scattering Angle $2\theta$')
p.ylabel('Normalized Intensity')
p.legend(['Simulated HCP Spectrum',
         'Ideal HCP Spectrum',])
p.title('Comparing Ideal HCP Spectrum to a Direct Simulation of ' \
        + str(n_cryst) + ' Crystallites')

p.figure()
p.plot(fcc_sim_angles[window], fcc_sim_norm_ints[window],'r')
p.plot(angles_ideal, fcc_norm_ideal, 'k--')
p.xlabel('Scattering Angle $2\theta$')
p.ylabel('Normalized Intensity')
p.legend(['Simulated FCC Spectrum',
         'Ideal FCC Spectrum',])
p.title('Comparing Ideal FCC Spectrum to a Direct Simulation of ' \
        + str(n_cryst) + ' Crystallites')


hcp_sim_small = hardcore_powder_XRD(sim.hcp_crystal,sim.wavelength,n_cryst,0.1, niceify=True)
hcp_sim_big = hardcore_powder_XRD(sim.hcp_crystal,sim.wavelength,n_cryst,2, niceify=True)
hcp_small_angles, hcp_small_intensities = spectrumify(hcp_sim_small)
hcp_big_angles, hcp_big_intensities = spectrumify(hcp_sim_big)
p.figure()
p.plot(hcp_small_angles, hcp_small_intensities / 0.1**3)
p.plot(hcp_sim_angles, hcp_sim_intensities / 0.5**3)
p.plot(hcp_big_angles, hcp_big_intensities / 2**3)
p.xlabel('Scattering Angle $2\theta$')
p.ylabel('Non-normalized Intensity (arbitrary units)')
p.legend(['0.1 um grains','0.5 um grains','2 um grains'])
p.title('Comparing Intensity of Simulated Spectra at Different Grain Sizes')



p.show()
