from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as p
from sim import *
from monte_carlo import *
from XRD_math import *


#
#
# This script just generates a lot of relevant graphs. It's
# designed to just make sure that all the different parts are
# working correctly.
#
#

wavelength = 2.253

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
sim = Sim((fcc_jet,hcp_jet),double_beam, wavelength, [0.5,0.5])

fcc_data, hcp_data = sim.sim(450,proof_plots=True)

angles_ideal, fcc_ideal, hcp_ideal, sum_ideal = mixture_graph(1)
hcp_norm_ideal = hcp_ideal / np.max(hcp_ideal)
fcc_norm_ideal = fcc_ideal / np.max(fcc_ideal)
p.figure()
p.plot(angles_ideal,hcp_ideal,'b-')
p.plot(angles_ideal,fcc_ideal,'r-')
p.xlabel(r'Scattering Angle $2\theta$')
p.ylabel('Intensity (arbitrary units)')
p.legend(['HCP Contribution','FCC Contribution'])
p.title('Simulated Ideal Spectrum of 1:1 HCP:FCC Mixture')


n_cryst = 100000

hcp_sim = monte_carlo_XRD(sim.hcp_crystal,sim.wavelength,n_cryst,0.5,
                              niceify=True)
fcc_sim = monte_carlo_XRD(sim.fcc_crystal,sim.wavelength,n_cryst,0.5,
                              niceify=True)
hcp_sim_angles, hcp_sim_intensities = spectrumify(hcp_sim)
fcc_sim_angles, fcc_sim_intensities = spectrumify(fcc_sim)
hcp_sim_norm_ints = hcp_sim_intensities / np.max(hcp_sim_intensities)
fcc_sim_norm_ints = fcc_sim_intensities / np.max(fcc_sim_intensities)
window = (hcp_sim_angles > 35) & (hcp_sim_angles < 55)
p.figure()
p.plot(hcp_sim_angles[window], hcp_sim_norm_ints[window],'b')
p.plot(angles_ideal, hcp_norm_ideal, 'k--')
p.xlabel(r'Scattering Angle $2\theta$')
p.ylabel('Normalized Intensity')
p.xlim([38,52])
p.legend(['Simulated HCP','Ideal HCP'])
#p.title('Comparing Ideal HCP Spectrum to a Direct Simulation of ' \
#        + str(n_cryst) + ' Crystallites')

p.figure()
p.plot(fcc_sim_angles[window], fcc_sim_norm_ints[window],'r')
p.plot(angles_ideal, fcc_norm_ideal, 'k--')
p.xlabel(r'Scattering Angle $2\theta$')
p.ylabel('Normalized Intensity')
p.legend(['Simulated FCC Spectrum',
         'Ideal FCC Spectrum',])
p.title('Comparing Ideal FCC Spectrum to a Direct Simulation of ' \
        + str(n_cryst) + ' Crystallites')


empty_jet = 0 * fcc_jet
full_jet = fcc_jet + hcp_jet
small_sim = Sim((empty_jet,full_jet),double_beam, wavelength, [0.125,0.125])
med_sim = Sim((empty_jet,full_jet),double_beam, wavelength, [0.25,0.25])
large_sim = Sim((empty_jet,full_jet),double_beam, wavelength, [0.5,0.5])


print(0)
small_hcp_data = small_sim.sim(550)[1]
med_hcp_data = med_sim.sim(550)[1]
large_hcp_data = large_sim.sim(550)[1]

small_hcp_data = {angle: [intensity] for angle, intensity
                  in small_hcp_data.items()}
med_hcp_data = {angle: [intensity] for angle, intensity
                  in med_hcp_data.items()}
large_hcp_data = {angle: [intensity] for angle, intensity
                  in large_hcp_data.items()}

for i in range(0,100):
    print(i+1)
    s = small_sim.sim(550)[1]
    m = med_sim.sim(550)[1]
    l = large_sim.sim(550)[1]
    for angle, intensity in s.items():
        small_hcp_data[angle] += [intensity]
    for angle, intensity in m.items():
        med_hcp_data[angle] += [intensity]
    for angle, intensity in l.items():
        large_hcp_data[angle] += [intensity]

p.figure()

small_hcp_angles, small_hcp_intensities = spectrumify(
    {angle: np.average(intensities) for angle, intensities
     in small_hcp_data.items()})
med_hcp_angles, med_hcp_intensities = spectrumify(
    {angle: np.average(intensities) for angle, intensities
     in med_hcp_data.items()})
large_hcp_angles, large_hcp_intensities = spectrumify(
    {angle: np.average(intensities) for angle, intensities
     in large_hcp_data.items()})


p.plot(large_hcp_angles,large_hcp_intensities, 'r-')
p.plot(med_hcp_angles,med_hcp_intensities, 'g-')
p.plot(small_hcp_angles,small_hcp_intensities, 'b-')
for angle, intensities in large_hcp_data.items():
    p.errorbar(angle, np.average(intensities), 
               yerr=np.std(intensities), fmt='r', capsize=10)
for angle, intensities in med_hcp_data.items():
    p.errorbar(angle, np.average(intensities), 
               yerr=np.std(intensities), fmt='g', capsize=10)
for angle, intensities in small_hcp_data.items():
    p.errorbar(angle, np.average(intensities), 
               yerr=np.std(intensities), fmt='b', capsize=10)
p.xlim([38,52])
#p.title('Scattering Intensities from a Pure HCP Jet')
p.legend(['0.5 um grains','0.25 um grains','0.125 um grains'], loc=2)
p.xlabel(r'Scattering Angle $2\theta$')
p.ylabel('Non-Normalized Intensity (arbitrary units)')



# angles, offsets, s_facts = monte_carlo_XRD(sim.hcp_crystal,
#                                                sim.wavelength,
#                                                n_cryst,0.1,
#                                                direct=True)

# p.figure()
# broad_ang, broad_int = gen_full_spectrum(angles,offsets,s_facts,0.1,sim.wavelength)
# p.plot(broad_ang,broad_int)
# p.title('Full Simulation Including Scherrer Broadening')
# p.xlabel(r'Scattering Angle $2\theta$')
# p.ylabel('Non-Normalized Intensity (arbitrary units)')


p.show()
