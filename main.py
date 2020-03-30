#!/usr/bin/env python3

from spectres import spectres
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from functools import lru_cache
import seaborn as sns; sns.set()

@lru_cache(maxsize=128)
def getflux(time, weather):
    data = pd.read_csv('spectra/' + weather + '/' + time + '.csv').to_numpy()
    flux = data[:,1]
    wl = data[:,0]

    flux = np.flip(flux)
    wl = np.flip(wl)
    return wl, flux

@lru_cache(maxsize=128)
def resample(wl, flux, SR, SNR):
    """Resample the spectrum with the given SR and SNR, and add the standard deviation."""
    wl = np.array(wl)
    flux = np.array(flux)
    newwl = np.linspace(3., 19., SR)
    newflux = spectres(newwl, wl, flux)

    N = SNR**2
    newflux *= N/np.amax(newflux)

    newfluxerr = np.sqrt(newflux)

    return newwl, newflux, newfluxerr

@lru_cache(maxsize=128)
def prepare(SR, SNR, time='modern', weather='clear'):
    """Get flux and resample"""
    wl, flux = getflux(time=time, weather=weather)
    wl, flux, err = resample(tuple(wl), tuple(flux), SR, SNR)
    return wl, flux, err


def sample_spectrum(flux):
    """Draw from a Poisson distribution to simulate the instrument's photon count."""
    sample = poisson.rvs(flux, size=(10000, flux.shape[0]))
    return sample

def logprob(hypothesis, data):
    """Return log of probability of the hypothesis, given the data."""
    logprobs = poisson.logpmf(data, hypothesis)
    total_logprob = np.sum(logprobs, axis=-1)
    total_logprob *= 2.7/10  # convert to log10 instead of ln
    return total_logprob


def find_ratio(flux1, flux2, sample):
    p1 = logprob(flux1, sample)
    p2 = logprob(flux2, sample)
    ratios = p1-p2
    # plt.hist(ratios)
    # plt.show()
    ratio = np.median(ratios)
    return ratio

def find_probmatrix(SR, SNR, times):
    Ntimes = len(times)
    probmatrix = np.zeros((Ntimes,Ntimes))

    for i, hypothesis_time in enumerate(times):
        for j, data_time in enumerate(times):
            _, H1_flux, _ = prepare(SR, SNR, time=hypothesis_time, weather='clear')
            _, H2_flux, _ = prepare(SR, SNR, time=data_time, weather='clear')
            data_flux = np.around(H2_flux)
            data_flux = sample_spectrum(data_flux)
            probmatrix[i,j] = find_ratio(H1_flux, H2_flux, data_flux)


    return probmatrix

def spectral_distance(probmatrix):
    distance = np.zeros(probmatrix.shape)
    n = probmatrix.shape[0]
    for i in range(n):
        for j in range(n):
            distance[i,j] = max(probmatrix[i,j], probmatrix[j,i])
            if i < j:
                distance[i,j] = np.nan
    return np.abs(distance)

weathers = ['clear', 'cloudy']
times = ['modern', '0.8Ga', '2.0Ga', '3.9Ga']


SR = 10
SNR = 10

probmatrix = find_probmatrix(SR, SNR, times)
distance = spectral_distance(probmatrix)


ax = sns.heatmap(distance, fmt='.1f', annot=True, cmap=sns.color_palette('Blues'))
ax.set_xticklabels(times)
ax.set_yticklabels(times)
# ax.set_xlabel('data')
# ax.set_ylabel('spectrum')
ax.set_title('metric distance between spectra\nSR = ' + str(SR) + ', SNR = ' + str(SNR))
# fig.tight_layout()
plt.show()
# plt.savefig('SR = ' + str(SR) + ', SNR = ' + str(SNR) +'.pdf')
# plt.close()
