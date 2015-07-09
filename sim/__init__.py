import numpy as np


class Sim(object):

    def __init__(self, jet_profile, beam_profile):
        self.fcc = jet_profile[0]
        self.hcp = jet_profile[1]
        self.beam = beam_profile
        
    def sim(self, offset):
        fcc_flat = np.sum(self.fcc,axis=1)
        hcp_flat = np.sum(self.hcp,axis=1)
        beam = self.beam[offset:offset+len(fcc_flat)]
        fcc = fcc_flat * beam.transpose()
        hcp = hcp_flat * beam.transpose()
        return fcc, hcp

