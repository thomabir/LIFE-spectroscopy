import numpy as np


class SpectrumSet:
    def __init__(self, spectra):
        self.spectra = spectra

    def resample(self, design):

        # given the peak SNR, calculate SNR of each individual spectrum
        peak_SNR = design.peak_SNR
        maxflux_i = [np.amax(spectrum.flux) for spectrum in self.spectra]
        maxflux = max(maxflux_i)
        SNR_i = np.sqrt(maxflux_i / maxflux) * peak_SNR

        # resample the spectra according to design and the calculated SNRs
        for SNR, spectrum in zip(SNR_i, self.spectra):
            spectrum.resample(lam_min=design.lam_min, lam_max=design.lam_max,
                              n_bins=design.n_bins, SNR=SNR)
