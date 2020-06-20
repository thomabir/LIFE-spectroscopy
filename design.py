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

        # n_bins and exptime are given:
        if self.SR is None and self.peak_SNR is None:
            self.SR = (lam_max + lam_min) / (lam_max - lam_min) * n_bins / 2
            self.peak_SNR = np.sqrt(exp_time / n_bins)

        # SR and SNR are given:
        if self.n_bins is None and self.exp_time is None:
            self.n_bins = 2 * SR * (lam_max - lam_min) / (lam_max + lam_min)
            self.n_bins = int( np.around(self.n_bins) )
            self.exp_time = peak_SNR**2 / self.n_bins
