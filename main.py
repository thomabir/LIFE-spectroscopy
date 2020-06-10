#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

from functools import lru_cache
from cached_property import cached_property  # pip install cached-property

import seaborn as sns
sns.set()
import itertools as it


from spectrum import Spectrum
from spectrum_set import SpectrumSet 
from inout import getflux
from ProbabilityModel import *
from design import Design


def find_utility(design, spectrum_set, samples):

    # generate actual hypothesis spectra from experimental design
    spectrum_set.resample(design)

    # calculate the utility
    pm = ProbabilityModel(spectrum_set=spectrum_set, samples=samples)
    pm.evaluate()
    utility, utility_std = pm.calculate_utility()

    # reset the cached memory (not necessary with this class structure)
    # for spectrum in spectrum_set.spectra:
    #    del spectrum.sample_spectrum

    return utility, utility_std

# set the scenarios
weathers = ['clear']  # , 'cloudy']
times = ['modern', '0.8Ga', '2.0Ga', '3.9Ga']
scenarios = it.product(times, weathers)


# load data from disk, and put into 'spectra'
spectra = []
for scenario in scenarios:
    wl, flux, label = getflux(scenario, convert_to_photons=True)
    new_spectrum = Spectrum(wl, flux, label)
    spectra.append(new_spectrum)


# to increase preformance for high-res original spectra,
# resample them with moderate resolution
for i, spectrum in enumerate(spectra):
    spectrum.resample(lam_min=2.9, lam_max=19.99, n_bins=1000, overwrite_original=True)

spectrum_set = SpectrumSet(spectra)



# set up experiment designs
exptimes = [100, 500, 1000, 2000, 4000]
all_n_bins = np.arange(2, 40, 1)
lam_max = 19.95
lam_min = 3.
samples = 100


for exptime in exptimes:
    all_SNR = np.sqrt(exptime / all_n_bins)    
    all_SR = (lam_max + lam_min)/(lam_max - lam_min) * all_n_bins / 2

    dist_low = np.zeros(all_SNR.shape)
    dist_high = np.zeros_like(dist_low)
    dist_med = np.zeros_like(dist_low)

    for i in range(all_n_bins.shape[0]):
        design = Design(n_bins = all_n_bins[i], exp_time=exptime, lam_min=lam_min, lam_max=lam_max)        
        U, err = find_utility(design, spectrum_set, samples)

        dist_low[i] = U - err
        dist_med[i] = U
        dist_high[i] = U + err


    plt.subplot(121)
    plt.fill_between(all_SR, dist_low, dist_high, label=exptime, alpha=0.3)
    plt.plot(all_SR, dist_med)

    plt.subplot(122)
    plt.fill_between(all_SNR, dist_low, dist_high, label=exptime, alpha=0.3)
    plt.plot(all_SNR, dist_med)
    

prior_entropy = -np.log(1/4)

plt.subplot(121)
plt.plot(all_SR, all_SR * 0 + prior_entropy, 'k')
plt.xlabel('SR')
plt.yscale('log')
#plt.xscale('log')
plt.ylabel('posterior entropy [nats]\n(lower is better)')
plt.title('Ideal SR for different exposure times')
#plt.legend()

plt.subplot(122)
plt.plot(np.linspace(4, 20, 2), prior_entropy*np.ones((2)), 'k', label='flat prior (baseline)')
plt.xlabel('SNR')
plt.yscale('log')
plt.title('Ideal SNR for different exposure times')
plt.legend()

plt.tight_layout()
plt.show()



