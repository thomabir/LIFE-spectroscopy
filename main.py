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
        self.fluxerr = 0
        self.label = label

        self.original_wl = wl
        self.original_flux = flux

    def resample(self, SR=10, SNR=100, overwrite_original=False):

        # features of original spectrum
        lam_min = max(np.amin(self.original_wl), 3.)  # we measure infrared, so lam >= 3 um
        lam_max = np.amax(self.original_wl)
        lam_avg = (lam_min + lam_max) / 2  # assumes uniform sampling

        # features of new spectrum
        delta_lam = lam_avg / SR
        lam_min += 3 * delta_lam
        lam_max -= 3 * delta_lam
        n_bins = (lam_max - lam_min) / delta_lam
        n_bins = int(n_bins)

        newwl = np.linspace(lam_min, lam_max, n_bins)
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

    def plot(self, *args, **kwargs):
        #plt.errorbar(self.wl, self.flux, yerr=self.fluxerr,
        #             label=self.label, fmt='o', markersize=8, capsize=8)
        plt.plot(self.wl, self.flux)
        plt.xlabel('wavelength [um]')
        plt.ylabel('photon count')
        plt.legend()


class ProbabilityModel:
    def __init__(self, spectra, samples=150):
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

                probmatrix[i, j, :] = logprob(hypothesis_flux, data_flux)
        
        self.p_x_d = np.copy(probmatrix)

        # normalize the log probability matrix.
        for j in range(self.n):

            # for numerical feasability, normalize such that pm[j,j] = 0.
            # this mathematically not necessary, but numerically important,
            # because the range of float64 would be exhausted pretty quickly.
            probmatrix[:, j, :] = probmatrix[:, j, :] - probmatrix[j, j, :]

            # normalize each row "properly", such that the probabilites sum up to 1.
            probmatrix[:, j, :] = probmatrix[:, j, :] - np.log(np.sum(np.e**probmatrix[:, j, :], axis=0))


        return probmatrix

    @cached_property
    def probmatr(self):
        return np.e**self.log_probmatr

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


def run_simulation(SR=10, SNR=100):
    for spectrum in spectra:
        spectrum.resample(SR=SR, SNR=SNR)

    samples = 50

    # find distance metric
    p = ProbabilityModel(spectra, samples=samples)
    pm = p.log_probmatr
    p_x_d = p.p_x_d

    newpm = np.zeros((4,4*samples))
    new_p_x_d = np.zeros((4, 4*samples))
    
    for i in range(4):
        newpm[i, :] = pm[i, :, :].flatten()
        new_p_x_d[i, :] = p_x_d[i, :, :].flatten()
    
    #print('entropy before:', -np.log(1/4))
    E_before = -np.log(1/4)

    P = newpm

    P_x_d = new_p_x_d

    P_x_d -= np.mean(P_x_d)    # normalize for better numerics
    P_x_d = np.sum(np.e**P_x_d, axis=0)
    P_x_d = P_x_d / np.sum(P_x_d) # normalize so that it sums to 1

    U_x_d = - P * np.exp(P)
    U_x_d = np.sum(U_x_d, axis=0)



    

    U_d = P_x_d * U_x_d
    U_d = np.sum(U_d)
    #print('entropy after:', U_d)
    E_after = U_d

    utility_gain = E_before - E_after

    for spectrum in spectra:
        del spectrum.sample_spectrum

    return E_after


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

def generate_heatmap(all_SNR=10, all_SR=100):
    dist = np.empty((len(all_SR), len(all_SNR)))

    for i, SR in enumerate(all_SR):
        for j, SNR in enumerate(all_SNR):
            # metric
            try:
                dist[i, j] = run_simulation(SR=SR, SNR=SNR)
            except:
                dist[i, j] = np.nan

    #title = '1 - confidence that most likely hypothesis is correct\n= "probability of being wrong"'
    title = '1 - difference between probabilities'
    plot_heatmap(dist, title)


from scipy.ndimage import median_filter

# to increase preformance for high-res original spectra,
# resample them with SR = 1000
for spectrum in spectra:
    spectrum.resample(SNR=None, SR=2000, overwrite_original=True)





exptimes = [500, 1000, 2000, 3000]
for exptime in exptimes:
    all_SR = np.geomspace(6, 1000, num=20)
    all_SNR = np.sqrt(exptime / all_SR)

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
        errs = np.zeros((5))
        for N in range(5):  # no of experiments, to determine error
            errs[N] = run_simulation(SR=all_SR[i], SNR=all_SNR[i])

        dist_low[i] = np.percentile(errs, 20)
        dist_med[i] = np.median(errs)
        dist_high[i] = np.percentile(errs, 80)

    plt.subplot(121)
    plt.fill_between(all_SR, dist_low, dist_high, label=exptime, alpha=0.2)
    plt.plot(all_SR, dist_med)
    plt.xlabel('SR')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('posterior entropy')
    plt.title('Ideal SR for different exposure times')
    plt.legend()

    plt.subplot(122)
    plt.fill_between(all_SNR, dist_low, dist_high, label=exptime, alpha=0.2)
    plt.plot(all_SNR, dist_med)
    plt.xlabel('SNR')
    plt.yscale('log')
    plt.ylabel('posterior entropy')
    plt.title('Ideal SNR for different exposure times')
    plt.legend()

plt.show()



