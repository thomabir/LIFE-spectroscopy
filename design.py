import numpy as np


class Design:
    def __init__(self, n_bins=None, lam_min=None, lam_max=None, SR=None,
                 peak_SNR=None,
                 exp_time=None):

        # spectral resolution
        self.SR = SR
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.n_bins = n_bins

        # signal to noise
        self.peak_SNR = peak_SNR

        # exposure time
        self.exp_time = exp_time

        # calculate (TODO implement solver for general cases)
        self.SR = (lam_max + lam_min) / (lam_max - lam_min) * n_bins / 2
        self.peak_SNR = np.sqrt(exp_time / n_bins)
