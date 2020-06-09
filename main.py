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


def convert(N_bins, N_photons):
    # to SR, peak SNR
    SNR = np.sqrt(N_photons)


class Spectrum:
    def __init__(self, wl, flux, label):
        self.wl = wl
        self.flux = flux
        self.fluxerr = 0
        self.label = label

        self.original_wl = wl
        self.original_flux = flux

    def resample(self, SR=None, SNR=None, n_bins=None, overwrite_original=False):

        # features of original spectrum
        lam_min = max(np.amin(self.original_wl), 3.)  # we measure infrared, so lam >= 3 um
        lam_max = min(np.amax(self.original_wl), 19.95)
        lam_avg = (lam_min + lam_max) / 2  # assumes uniform sampling

        if overwrite_original:
            lam_min = 1.1
            lam_max = 19.99



        # calculate n_bins from SR, if necessary
        if SR:
            delta_lam = lam_avg / SR
            # lam_min += 1 * delta_lam
            # lam_max -= 1 * delta_lam
            n_bins = (lam_max - lam_min) / delta_lam
            n_bins = int(n_bins)
            corners = np.linspace(lam_min, lam_max, n_bins+1)
            midpoints = (corners[1:] + corners[:-1]) / 2
        else:
            corners = np.linspace(lam_min, lam_max, n_bins+1)
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

        # print('N_tot', np.sum(self.flux), 'SNR', SNR, 'n_bins', n_bins)

    @cached_property
    def sample_spectrum(self):
        samples = self.samples
        flux = self.flux
        return poisson.rvs(flux, size=(samples, flux.shape[0]))

    def plot(self, *args, **kwargs):
        plt.errorbar(self.wl, self.flux, yerr=self.fluxerr,
                    label=self.label, fmt='o', markersize=8, capsize=8)
        # plt.plot(self.wl, self.flux)
        plt.xlabel('wavelength [um]')
        plt.ylabel('photon count')
        plt.legend()


class ProbabilityModel:
    def __init__(self, spectra, samples=100):
        self.spectra = spectra
        self.n = len(spectra)
        self.samples = samples

        for spectrum in self.spectra:
            spectrum.samples = samples

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

            return total_logprob

        probmatrix = np.zeros((self.n, self.n, self.samples))

        # compute the unnormalized log probability matrix
        for i, hypothesis_spectrum in enumerate(spectra):
            for j, data_spectrum in enumerate(spectra):

                hypothesis_flux = hypothesis_spectrum.flux
                data_flux = data_spectrum.sample_spectrum

                # p(x|H)
                probmatrix[i, j, :] = logprob(hypothesis_flux, data_flux)

        # p(x|H)
        self.p_x_H = np.copy(probmatrix)

        # normalize the log probability matrix.
        for j in range(self.n):

            # for numerical feasability, normalize such that pm[j,j] = 0.
            # this is mathematically not necessary, but numerically important,
            # because the range of float64 would be exhausted pretty quickly.
            probmatrix[:, j, :] = probmatrix[:, j, :] - probmatrix[j, j, :]


            # normalize each row "properly", such that the probabilites sum up to 1.
            probmatrix[:, j, :] = probmatrix[:, j, :] - np.log(np.sum(np.e**probmatrix[:, j, :], axis=0))

        # probmatrix = np.median(probmatrix, axis=-1)
        # p(H|x)
        return probmatrix

    @cached_property
    def probmatr(self):
        return np.e**self.log_probmatr

    @cached_property
    def metric1(self):
        n = self.n
        pm = self.probmatr
        pm = np.median(pm, axis=-1)
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
# times = ['0.8Ga', '2.0Ga',]

scenarios = it.product(times, weathers)

# load data from disk, and put into 'spectra'
spectra = []
for scenario in scenarios:
    wl, flux, label = getflux(scenario, convert_to_photons=True)
    new_spectrum = Spectrum(wl, flux, label)
    spectra.append(new_spectrum)


def run_simulation(SR=None, SNR=None, n_bins=None):
    SNR_max = SNR
    # determine the largest flux in all spectra
    maxflux_i = [np.amax(spectrum.flux) for spectrum in spectra]
    maxflux = max(maxflux_i)


    SNR_i = np.sqrt(maxflux_i / maxflux) * SNR

    for SNR, spectrum in zip(SNR_i, spectra):
        spectrum.resample(SR=SR, SNR=SNR, n_bins=n_bins)
    #     plt.title(str(SNR) + ' ' + str(SR))
    #     spectrum.plot()

    # plt.show()

    # add noise
    # factor = n_bins * SNR**2 / 500  # propto exptime
    # for spectrum in spectra:
    #     spectrum.flux += 0.5*factor#poisson.rvs(10, size=(spectrum.flux.shape[0]))

    samples = 100

    # find distance metric
    p = ProbabilityModel(spectra, samples=samples)
    log_p_H_x = p.log_probmatr


    log_p_x_H = p.p_x_H

    # reshape
    new_log_p_H_x = np.zeros((4,4*samples))
    new_log_p_x_H = np.zeros((4,4*samples))
    
    for i in range(4):
        new_log_p_H_x[i, :] = log_p_H_x[i, :, :].flatten()
        new_log_p_x_H[i, :] = log_p_x_H[i, :, :].flatten()
    
    #print('entropy before:', -np.log(1/4))
    E_before = -np.log(1/4)

    log_p_H_x = new_log_p_H_x

    log_p_x_H = new_log_p_x_H

    # normalize and sum
    log_p_x_H -= np.mean(log_p_x_H)    # normalize for better numerics
    p_x = np.sum(np.e**log_p_x_H, axis=0)
    p_x = p_x / np.sum(p_x) # normalize so that it sums to 1

    U_x = - log_p_H_x * np.exp(log_p_H_x)
    U_x = np.sum(U_x, axis=0)

    # p_x = np.ones(p_x.shape) / p_x.shape[0]
    # plt.subplot(131)
    # plt.hist(U_x, label='U_x')
    # plt.subplot(132)
    # plt.hist(p_x, label='p_x')
    # plt.subplot(133)
    # plt.hist(U_x*p_x, label='p_x * U_x')
    # plt.legend()
    # plt.show()
    # exit()

    
    # U = np.sum(p_x * U_x)

    # monte carlo estimator:
    U = np.mean(U_x)


    #print('entropy after:', U_d)
    E_after = U

    utility_gain = E_before - E_after

    # print(SNR, n_bins)
    # print(p.probmatr[:, :, 0])
    # print(np.median(U_x))
    # print(np.sum(p_x), np.median(p_x), np.mean(p_x))
    # print()

    err = np.std(U_x)/np.sqrt(samples)

    for spectrum in spectra:
        del spectrum.sample_spectrum

    return U, err

def run_other_simulation(SR=None, SNR=None, n_bins=None):
    SNR_max = SNR
    # determine the largest flux in all spectra
    maxflux_i = [np.amax(spectrum.flux) for spectrum in spectra]
    maxflux = max(maxflux_i)


    SNR_i = np.sqrt(maxflux_i / maxflux) * SNR

    for SNR, spectrum in zip(SNR_i, spectra):
        spectrum.resample(SR=SR, SNR=SNR, n_bins=n_bins)
    #     plt.title(str(SNR) + ' ' + str(SR))
    #     spectrum.plot()

    # plt.show()

    # add noise
    # factor = n_bins * SNR**2 / 500  # propto exptime
    # for spectrum in spectra:
    #     spectrum.flux += 0.5*factor#poisson.rvs(10, size=(spectrum.flux.shape[0]))

    samples = 500

    # find distance metric
    p = ProbabilityModel(spectra, samples=samples)
    result = 1 - p.metric1

    for spectrum in spectra:
        del spectrum.sample_spectrum
    return result, 0





# to increase preformance for high-res original spectra,
# resample them with SR = 1000
for i, spectrum in enumerate(spectra):
    spectrum.resample(SNR=None, n_bins=1000, overwrite_original=True)
    # spectrum.flux = spectrum.flux * 0 + i + 1
    # spectrum.plot()
    # plt.show()



exptimes = [500, 1000, 2000, 4000]
for exptime in exptimes:
    all_n_bins = np.arange(2, 40, 1)
    #print(all_n_bins)
    all_SNR = np.sqrt(exptime / all_n_bins)

    # convert to SR
    lam_max = 19.95
    lam_min = 3.
    all_SR = (lam_max + lam_min)/(lam_max - lam_min) * all_n_bins / 2


    # all_SNR = np.geomspace(0.1,20, num=20)
    # all_SR = exptime / all_SNR**2
    # idx = (5 < all_SR) & (all_SR < 200)
    # all_SNR = all_SNR[idx]
    # all_SR = all_SR[idx]
    # print(all_SR)

    dist_low = np.zeros(all_SNR.shape)
    dist_high = np.zeros_like(dist_low)
    dist_med = np.zeros_like(dist_low)


    for i in range(all_SNR.shape[0]):
        U_x, err = run_other_simulation(n_bins=all_n_bins[i], SNR=all_SNR[i])

        # if i==0:
        # for spectrum in spectra:
        #     spectrum.plot()
        # plt.title('SNR:' + str(all_SNR[i]) + 'bins:' +  str(all_n_bins[i]))
        # plt.show()
        # exit()

        dist_low[i] = U_x - err
        dist_med[i] = U_x
        dist_high[i] = U_x + err


    # avg = np.mean(dist_med)
    # dist_low /= avg
    # dist_med /= avg
    # dist_high /= avg

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



