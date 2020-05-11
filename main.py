#!/usr/bin/env python3

from spectres import spectres
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

from functools import lru_cache
from cached_property import cached_property  # pip install cached-property

import seaborn as sns
sns.set()
import itertools as it


@lru_cache(maxsize=128)
def getflux(scenario, convert_to_photons=True):

    time = scenario[0]
    weather = scenario[1]

    data = pd.read_csv('spectra/' + weather + '/' + time + '.csv').to_numpy()
    flux = data[:, 1]
    wl = data[:, 0]

    flux = np.flip(flux)
    wl = np.flip(wl)

    # E propto N / lambda
    # convert from energy to photons
    if convert_to_photons:
        flux = flux * wl

    label = time + ' ' + weather

    return wl, flux, label


class Spectrum:
    def __init__(self, wl, flux, label):
        self.wl = wl
        self.flux = flux
        self.label = label

        self.original_wl = wl
        self.original_flux = flux

    def resample(self, SR, SNR):
        lam_min = 3.
        lam_max = 19.
        lam_avg = (lam_max + lam_min) / 2

        delta_lam = lam_avg / SR
        n_bins = (lam_max - lam_min) / delta_lam
        n_bins = int(n_bins)

        newwl = np.linspace(3., 19., n_bins)
        newflux = spectres(newwl, self.original_wl, self.original_flux)

        N = SNR**2
        newflux *= N / np.amax(newflux)

        newfluxerr = np.sqrt(newflux)

        self.wl = newwl
        self.flux = newflux
        self.fluxerr = newfluxerr

    @cached_property
    def sample_spectrum(self, N=1000):
        flux = np.around(self.flux)
        return poisson.rvs(flux, size=(N, flux.shape[0]))

    def plot(self, *args, **kwargs):
        plt.errorbar(self.wl, self.flux, yerr=self.fluxerr,
                     label=self.label, fmt='o', markersize=8, capsize=8)
        plt.xlabel('wavelength [um]')
        plt.ylabel('photon count')
        plt.legend()


class ProbabilityModel:
    def __init__(self, spectra):
        self.spectra = spectra
        self.n = len(spectra)

    @cached_property
    def log_probmatr(self):
        """
        Returns
        logprob[i,j]:
            i ... hypothesis
            j ... data
        """

        def logprob(hypothesis, data):
            """Return median log_prob of the hypothesis, given the data."""

            # for each data point in each sampled spectrum
            logprobs = poisson.logpmf(data, hypothesis)

            # for each sampled spectrum
            total_logprob = np.sum(logprobs, axis=-1)
            total_logprob *= np.e / 10  # convert to log10 instead of ln

            # average over all sampled spectra
            # median is used since it is almost invariant under log
            avg_prob = np.median(total_logprob)
            return avg_prob

        probmatrix = np.zeros((self.n, self.n))

        # compute the unnormalized log probability matrix
        for i, hypothesis_spectrum in enumerate(spectra):
            for j, data_spectrum in enumerate(spectra):

                hypothesis_flux = hypothesis_spectrum.flux
                data_flux = data_spectrum.sample_spectrum

                probmatrix[i, j] = logprob(hypothesis_flux, data_flux)

        # normalize the log probability matrix.
        for j in range(self.n):

            # for numerical feasability, normalize such that pm[j,j] = 0.
            # this mathematically not necessary, but numerically important,
            # because the range of float64 would be exhausted pretty quickly.
            probmatrix[:, j] = probmatrix[:, j] - probmatrix[j, j]

            # normalize each row "properly", such that the probabilites sum up to 1.
            probmatrix[:, j] = probmatrix[:, j] - np.log10(np.sum(10**probmatrix[:, j]))

        return probmatrix

    @cached_property
    def probmatr(self):
        return 10**self.log_probmatr

    @cached_property
    def metric1(self):
        n = self.n
        pm = self.probmatr
        distance = np.zeros(pm.shape)

        for i in range(self.n):
            for j in range(self.n):
                if i > j:
                    d1 = abs(pm[i, j] - pm[j, j])
                    d2 = abs(pm[j, i] - pm[i, i])
                    distance[i, j] = min(d1, d2)
                if i <= j:
                    distance[i, j] = np.nan
        return np.nanmin(distance)

    @cached_property
    def metric2(self):
        n = self.n
        pm = self.log_probmatr
        distance = np.zeros(pm.shape)

        for i in range(self.n):
            for j in range(self.n):
                if i > j:
                    d1 = abs(pm[i, j] - pm[j, j])
                    d2 = abs(pm[j, i] - pm[i, i])
                    distance[i, j] = min(d1, d2)
                if i <= j:
                    distance[i, j] = np.nan

        return np.nanmin(distance)

    @cached_property
    def metric3(self):
        """Confidence of most likely hypothesis"""
        n = self.n
        pm = self.probmatr


        # pm[hypothesis, data]
        conf = np.nanmax(pm, axis=0)
        return np.amin(conf)


# def plot_probmatrix(matrix):
#     ax = sns.heatmap(matrix, fmt='.1f', annot=True,
#                      cmap=sns.color_palette('Blues_r'))
#     ax.set_xlabel('data')
#     ax.set_ylabel('hypothesis')
#     ax.set_xticklabels(times)
#     ax.set_yticklabels(times)
#     ax.set_title('log Probability of hypothesis, given data\nSR = ' +
#                  str(SR) + ', SNR = ' + str(SNR))
#     # fig.tight_layout()
#     plt.show()
#     # plt.savefig('probabilities-SR=' + str(SR) + '-SNR=' + str(SNR) + '.pdf')
#     # plt.close()


# def plot_distance(matrix):
#     ax = sns.heatmap(distance, fmt='.5f', annot=True,
#                      cmap=sns.color_palette('Blues'))
#     ax.set_xticklabels(times)
#     ax.set_yticklabels(times)
#     # ax.set_xlabel('data')
#     # ax.set_ylabel('spectrum')
#     ax.set_title('metric distance between spectra\nSR = ' +
#                  str(SR) + ', SNR = ' + str(SNR))
#     # fig.tight_layout()
#     plt.show()
#     # plt.savefig('difference-ratio-SR=' + str(SR) + '-SNR=' + str(SNR) + '.pdf')
#     # plt.close()


weathers = ['clear']  # , 'cloudy']
times = ['modern', '0.8Ga', '2.0Ga', '3.9Ga']

scenarios = it.product(times, weathers)

# load data from disk, and put into 'spectra'
spectra = []
for scenario in scenarios:
    wl, flux, label = getflux(scenario, convert_to_photons=True)
    new_spectrum = Spectrum(wl, flux, label)
    spectra.append(new_spectrum)


def run_simulation(SR, SNR):
    for spectrum in spectra:
        spectrum.resample(SR, SNR)
        try:
            del spectrum.sample_spectrum
        except:
            pass
        # spectrum.plot()
    # plt.show()

    # find distance metric
    p = ProbabilityModel(spectra)
    result = p.metric1
    return 1 - result


def plot_heatmap(dist, title):
    ax = sns.heatmap(dist,
                     # fmt='.1f',
                     annot=True,
                     cmap=sns.color_palette('Blues_r'),
                     )
    ax.set_xlabel('SNR')
    ax.set_ylabel('SR')
    ax.set_xticklabels(np.around(all_SNR))
    ax.set_yticklabels(np.around(all_SR))
    ax.set_title(title)
    plt.savefig('metric1.pdf')
    plt.show()

# all_SR = [20, 100, 1000]
all_SNR = [2, 3, 5, 7, 10, 20]

#N_tot = SNR**2 * SR
#all_SR = 500 / np.array(all_SNR)**2
all_SR = [10, 20, 50, 100]
# all_SNR = np.sqrt(2000 / np.array(all_SR))



dist = np.empty((len(all_SR), len(all_SNR)))

for i, SR in enumerate(all_SR):
    for j, SNR in enumerate(all_SNR):
        # metric
        try:
            dist[i, j] = run_simulation(SR, SNR)
        except:
            dist[i, j] = np.nan

#title = '1 - confidence that most likely hypothesis is correct\n= "probability of being wrong"'
title = '1 - difference between probabilities'
plot_heatmap(dist, title)


