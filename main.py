#!/usr/bin/env python3

from spectres import spectres
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from functools import lru_cache
import seaborn as sns; sns.set()
import itertools as it

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

    lam_min = 3.
    lam_max = 19.
    lam_avg = (lam_max - lam_min)/2

    delta_lam = lam_avg/SR
    n_bins = (lam_max-lam_min)/delta_lam
    n_bins = int(n_bins)

    newwl = np.linspace(3., 19., n_bins)
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


def find_probability(hypothesis_flux, data_flux):
    probs = logprob(hypothesis_flux, data_flux)
    # ratios = p1-p2
    # plt.hist(ratios)
    # plt.show()
    prob = np.median(probs)
    return prob

# plt.rcParams.update({'lines.markeredgewidth': 1})

def find_probmatrix(SR, SNR, times):
    Ntimes = len(times)
    probmatrix = np.zeros((Ntimes, Ntimes))

    for i, hypothesis_time in enumerate(times):
        for j, data_time in enumerate(times):
            _, hypothesis_flux, _ = prepare(SR, SNR, time=hypothesis_time, weather='clear')
            wl, data_flux, _ = prepare(SR, SNR, time=data_time, weather='clear')
            data_flux = np.around(data_flux)
            data_flux = sample_spectrum(data_flux)
            probmatrix[i,j] = find_probability(hypothesis_flux, data_flux)

    for j, _ in enumerate(times):
        probmatrix[:,j] = probmatrix[:,j]-np.log10( np.sum(10**probmatrix[:,j]) )

    return probmatrix

def spectral_distance(probmatrix):
    distance = np.zeros(probmatrix.shape)
    n = probmatrix.shape[0]
    for i in range(n):
        for j in range(n):
            # distance[i,j] = np.amax([probmatrix[i,j]-probmatrix[j,j], probmatrix[j,i]-probmatrix[i,i]])
            distance[i,j] = 10**probmatrix[i,j] - 10**probmatrix[j,j]
            if i < j:
                distance[i,j] = np.nan
    return np.abs(distance)

def plot_spectra(SR, SNR, scenarios):
    for (time, weather) in scenarios:
        wl, flux, err = prepare(SR, SNR, time=time, weather=weather)
        plt.errorbar(wl, flux, yerr=err, label=time+' '+weather, fmt='o', markersize=8, capsize=8)
    plt.xlabel('wavelength [um]')
    plt.ylabel('photon count')
    plt.title('Earth spectra at different times\nSR = '+str(SR)+', peak SNR = '+str(SNR))
    plt.legend()
    # plt.show()
    plt.savefig('spectra.pdf')
    plt.close()

class SpectroscopyModel:
    def __init__(self, SR, SNR, scenarios):
        self.SR = SR
        self.SNR = SNR
        self.scenarios = scenarios

    def get_spectrum(self):
        pass

    def probmatr(self):
        pass

    def log_probmatr(self):
        pass

    def metric1(self):
        pass

    def metric2(self):
        pass

weathers = ['clear']#, 'cloudy']
times = ['modern', '0.8Ga', '2.0Ga', '3.9Ga']

scenarios = it.product(times, weathers)

SR = 10
SNR = 10

plot_spectra(SR, SNR, scenarios)

# M = SpectroscopyModel(SR, SNR, scenarios)
# M.probmatr.plot()
# M.log_probmatr.plot()
# M.metric1.plot()
# M.metric2.plot()

probmatrix = find_probmatrix(SR, SNR, times)

def plot_probmatrix(matrix):
    ax = sns.heatmap(matrix, fmt='.1f', annot=True, cmap=sns.color_palette('Blues_r'))
    ax.set_xlabel('data')
    ax.set_ylabel('hypothesis')
    ax.set_xticklabels(times)
    ax.set_yticklabels(times)
    ax.set_title('log Probability of hypothesis, given data\nSR = ' + str(SR) + ', SNR = ' + str(SNR))
    # fig.tight_layout()
    # plt.show()
    plt.savefig('probabilities-SR=' + str(SR) + '-SNR=' + str(SNR) +'.pdf')
    plt.close()

def plot_distance(matrix):
    ax = sns.heatmap(distance, fmt='.5f', annot=True, cmap=sns.color_palette('Blues'))
    ax.set_xticklabels(times)
    ax.set_yticklabels(times)
    # ax.set_xlabel('data')
    # ax.set_ylabel('spectrum')
    ax.set_title('metric distance between spectra\nSR = ' + str(SR) + ', SNR = ' + str(SNR))
    # fig.tight_layout()
    # plt.show()
    plt.savefig('difference-ratio-SR=' + str(SR) + '-SNR=' + str(SNR) +'.pdf')
    plt.close()

plot_probmatrix(probmatrix)

distance = spectral_distance(probmatrix)
plot_distance(probmatrix)
