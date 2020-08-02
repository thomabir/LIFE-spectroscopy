#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

from functools import lru_cache
from cached_property import cached_property  # pip install cached-property

import seaborn as sns
#sns.set()
import itertools as it


from spectrum import Spectrum
from spectrum_set import SpectrumSet 
from inout import getflux
from ProbabilityModel import *
from design import Design


def find_utility(design, spectrum_set, samples, utility=None):

    # generate actual hypothesis spectra from experimental design
    spectrum_set.resample(design)

    # calculate the utility
    pm = ProbabilityModel(spectrum_set=spectrum_set, samples=samples)
    pm.evaluate()
    utility, utility_std = pm.calculate_utility(utility=utility)

    # reset the cached memory (not necessary with this class structure)
    # for spectrum in spectrum_set.spectra:
    #    del spectrum.sample_spectrum

    return utility, utility_std

def plot_heatmap(dist, all_SNR, all_SR, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(columnwidth, smallfigheight))
    ax = sns.heatmap(dist,
                     # fmt='.1f',
                     annot=True,
                     cmap=sns.color_palette('Blues_r'),
                     cbar=False
                     )
    ax.set_xlabel('SNR')
    ax.set_ylabel('SR')
    ax.set_xticklabels(np.around(all_SNR))
    ax.set_yticklabels(np.around(all_SR))
    # ax.set_title(title)
    plt.savefig('fig/' + filename + '-heatmap.pdf', bbox_inches='tight')


# set the scenarios
weathers = ['clear' , 'cloudy']
times = ['0.0', '0.8', '2.0', '3.9']
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
    spectrum.resample(lam_min=4.05, lam_max=19.99, n_bins=1000, overwrite_original=True)


spectrum_set = SpectrumSet(spectra)






# set up experiment designs
exptimes = [400, 1000, 4000]
all_n_bins = np.arange(2, 200, 1)
lam_min = 4.1
lam_max = 19.9
samples = 500



plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rc('legend', fontsize=7)


textwidth = 7.02893
figheight = 3
smallfigheight = 2.5
columnwidth = 3.44527



# plot some examples

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
    gridspec_kw={'hspace': 0, 'wspace': 0},
    figsize=(textwidth, figheight))

# design = Design(SR=5, peak_SNR=15, lam_min=lam_min, lam_max=lam_max)
#design = Design(SR=30, peak_SNR=7, lam_min=lam_min, lam_max=lam_max)
#spectrum_set.resample(design)
for spectrum in spectrum_set.spectra:
    if 'clear' in spectrum.label:
        ax1.plot(spectrum.wl, spectrum.flux/300, linewidth=0.4, label=spectrum.label[:3])
        ax1.set_xlabel('wavelength [µm]')
        ax1.set_ylabel('normalized photon count')
    if 'cloudy' in spectrum.label:
        ax2.plot(spectrum.wl, spectrum.flux/300, linewidth=0.4, label=spectrum.label[:3])
        ax2.set_xlabel('wavelength [µm]')
        #ax2.set_ylabel('photon count')

ax1.set_title('clear weather')
ax2.set_title('cloudy weather')

ax2.legend(title='time ago [billion years]', loc='upper left', frameon=False)


plt.savefig('fig/earth-spectra.pdf', bbox_inches='tight')



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
    gridspec_kw={'hspace': 0, 'wspace': 0},
    figsize=(textwidth, figheight))


# run experiment: entropy
for j, exptime in enumerate(exptimes):
    all_SNR = np.sqrt(exptime / all_n_bins)    
    all_SR = (lam_max + lam_min)/(lam_max - lam_min) * all_n_bins / 2


    dist_low = np.zeros(all_SNR.shape)
    dist_high = np.zeros_like(dist_low)
    dist_med = np.zeros_like(dist_low)

    for i in range(all_n_bins.shape[0]):
        design = Design(n_bins = all_n_bins[i], exp_time=exptime, lam_min=lam_min, lam_max=lam_max)        
        U, err = find_utility(design, spectrum_set, samples, utility='information')

        dist_low[i] = U - err
        dist_med[i] = U
        dist_high[i] = U + err

    ax1.fill_between(all_SR, dist_low, dist_high, label=exptime, alpha=0.3)
    ax1.plot(all_SR, dist_med)

    ax2.fill_between(all_SNR, dist_low, dist_high, label=exptime, alpha=0.3)
    ax2.plot(all_SNR, dist_med)


prior_entropy = - np.log2(1/4)

ax1.plot(all_SR, all_SR * 0 + prior_entropy, 'k', label='0 (flat prior, baseline)')
ax1.set_xlabel('SR')
ax1.set_yscale('log')
ax1.set_ylabel('posterior entropy [bits]\n(lower is better)')


ax2.plot(np.linspace(1.4, 45, 2), prior_entropy*np.ones((2)), 'k', label='0 (flat prior, baseline)')
ax2.set_xlabel('SNR')
ax2.set_yscale('log')
ax2.legend(title='exposure time', loc='upper right')


plt.savefig('fig/entropy-exptime.pdf', bbox_inches='tight')





fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
    gridspec_kw={'hspace': 0, 'wspace': 0},
    figsize=(textwidth, figheight))

# run experiment: confidence
for exptime in exptimes:
    all_SNR = np.sqrt(exptime / all_n_bins)    
    all_SR = (lam_max + lam_min)/(lam_max - lam_min) * all_n_bins / 2


    dist_low = np.zeros(all_SNR.shape)
    dist_high = np.zeros_like(dist_low)
    dist_med = np.zeros_like(dist_low)

    for i in range(all_n_bins.shape[0]):
        design = Design(n_bins = all_n_bins[i], exp_time=exptime, lam_min=lam_min, lam_max=lam_max)        
        U, err = find_utility(design, spectrum_set, samples, utility='confidence')

        dist_low[i] = U - err
        dist_med[i] = U
        dist_high[i] = U + err


    ax1.fill_between(all_SR, dist_low, dist_high, label=exptime, alpha=0.3)
    ax1.plot(all_SR, dist_med)

    ax2.fill_between(all_SNR, dist_low, dist_high, label=exptime, alpha=0.3)
    ax2.plot(all_SNR, dist_med)


prior_likelihood = 1 - 1/4

ax1.plot(all_SR, all_SR * 0 + prior_likelihood, 'k', label='0 (flat prior, baseline)')
ax1.set_xlabel('SR')
ax1.set_yscale('log')
ax1.set_ylabel('1 - confidence of inference\n(lower is better)')

ax2.plot(np.linspace(1.4, 45, 2), prior_likelihood*np.ones((2)), 'k')
ax2.set_xlabel('SNR')
ax2.set_yscale('log')

plt.savefig('fig/likelihood-exptime.pdf', bbox_inches='tight')









fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
    gridspec_kw={'hspace': 0, 'wspace': 0},
    figsize=(textwidth, figheight))

# run experiment: probability
for exptime in exptimes:
    all_SNR = np.sqrt(exptime / all_n_bins)    
    all_SR = (lam_max + lam_min)/(lam_max - lam_min) * all_n_bins / 2


    dist_low = np.zeros(all_SNR.shape)
    dist_high = np.zeros_like(dist_low)
    dist_med = np.zeros_like(dist_low)

    for i in range(all_n_bins.shape[0]):
        design = Design(n_bins = all_n_bins[i], exp_time=exptime, lam_min=lam_min, lam_max=lam_max)        
        U, err = find_utility(design, spectrum_set, samples, utility='wrong')

        dist_low[i] = U - err
        dist_med[i] = U
        dist_high[i] = U + err


    ax1.fill_between(all_SR, dist_low, dist_high, label=exptime, alpha=0.3)
    ax1.plot(all_SR, dist_med)

    ax2.fill_between(all_SNR, dist_low, dist_high, label=exptime, alpha=0.3)
    ax2.plot(all_SNR, dist_med)


prior_likelihood = 1 - 1/4

ax1.set_ylim(bottom=1e-4)
ax2.set_ylim(bottom=1e-4)

ax1.plot(all_SR, all_SR * 0 + prior_likelihood, 'k', label='0 (flat prior, baseline)')
ax1.set_xlabel('SR')
ax1.set_yscale('log')
ax1.set_ylabel('probability of inference not\nagreeing with ground truth')

ax2.plot(np.linspace(1.4, 45, 2), prior_likelihood*np.ones((2)), 'k')
ax2.set_xlabel('SNR')
ax2.set_yscale('log')

plt.savefig('fig/probability-exptime.pdf', bbox_inches='tight')







#
# heatmaps
#

all_SNR = [1, 5, 10, 20]
all_SR = [10, 50, 100]


dist = np.empty((len(all_SR), len(all_SNR)))

for i, SR in enumerate(all_SR):
    for j, SNR in enumerate(all_SNR):
        design = Design(SR = SR, peak_SNR=SNR, lam_min=lam_min, lam_max=lam_max)        
        dist[i, j], _ = find_utility(design, spectrum_set, samples, utility='information')

plot_heatmap(dist, all_SNR, all_SR, 'posterior entropy', 'entropy')


for i, SR in enumerate(all_SR):
    for j, SNR in enumerate(all_SNR):
        design = Design(SR = SR, peak_SNR=SNR, lam_min=lam_min, lam_max=lam_max)        
        dist[i, j], _ = find_utility(design, spectrum_set, samples, utility='confidence')

plot_heatmap(dist, all_SNR, all_SR, '1 - confidence of inference', 'likelihood')



for i, SR in enumerate(all_SR):
    for j, SNR in enumerate(all_SNR):
        design = Design(SR = SR, peak_SNR=SNR, lam_min=lam_min, lam_max=lam_max)        
        dist[i, j], _ = find_utility(design, spectrum_set, samples, utility='wrong')

plot_heatmap(dist, all_SNR, all_SR, 'probability of wrong inference', 'probability')