from spectres import spectres
import numpy as np
from cached_property import cached_property  # pip install cached-property
import matplotlib.pyplot as plt
from scipy.stats import poisson


class Spectrum:
    def __init__(self, wl, flux, label):
        self.wl = wl
        self.flux = flux
        self.fluxerr = 0
        self.label = label

        self.original_wl = wl
        self.original_flux = flux

    def resample(self, lam_min=None, lam_max=None, n_bins=None,
                 SNR=None, overwrite_original=False):

        corners = np.linspace(lam_min, lam_max, n_bins + 1)
        midpoints = (corners[1:] + corners[:-1]) / 2

        newwl = midpoints
        newflux = spectres(newwl, self.original_wl, self.original_flux)

        if SNR:
            N = SNR**2      # no. of photons per bin
            newflux *= N / np.amax(newflux)
            self.fluxerr = np.sqrt(newflux)

        if overwrite_original:
            self.original_wl = newwl
            self.original_flux = newflux

        self.wl = newwl
        self.flux = newflux

    @cached_property
    def sample_spectrum(self):
        samples = self.samples
        flux = self.flux
        return poisson.rvs(flux, size=(samples, flux.shape[0]))

    def plot(self, ax, mode='discrete', *args, **kwargs):
        if mode == 'discrete':
            ax.errorbar(self.wl, self.flux, yerr=self.fluxerr,
                        label=self.label, fmt='o', markersize=8, capsize=8)

        if mode == 'continuous':
            ax.fill_between(self.wl, self.flux - self.fluxerr,
                            self.flux + self.fluxerr,
                            alpha=0.2)
            ax.plot(self.wl, self.flux, label=self.label)

        # plt.plot(self.wl, self.flux)
        ax.set_xlabel('wavelength [um]')
        ax.set_ylabel('photon count')
        ax.legend()
