import numpy as np
from scipy.stats import poisson

from cached_property import cached_property  # pip install cached-property

from spectrum import *


class ProbabilityModel:
    def __init__(self, spectrum_set=None, samples=100):
        self.spectrum_set = spectrum_set
        self.spectra = spectrum_set.spectra
        self.n = len(self.spectra)
        self.samples = samples

        for spectrum in self.spectra:
            spectrum.samples = samples

        self.utility_function = None

    def evaluate(self):

        def logprob(hypothesis, data):
            """Return log_prob of the hypothesis, given the data."""
            # for each data point in each sampled spectrum
            logprobs = poisson.logpmf(data, hypothesis)

            # for each sampled spectrum
            total_logprob = np.sum(logprobs, axis=-1)

            return total_logprob

        probmatrix = np.zeros((self.n, self.n, self.samples))

        # compute the unnormalized log probability matrix
        for i, hypothesis_spectrum in enumerate(self.spectra):
            for j, data_spectrum in enumerate(self.spectra):

                hypothesis_flux = hypothesis_spectrum.flux
                data_flux = data_spectrum.sample_spectrum

                # p(x|H)
                probmatrix[i, j, :] = logprob(hypothesis_flux, data_flux)

        # p(x|H)
        self.log_p_x_H = np.copy(probmatrix)

        # normalize the log probability matrix.
        for j in range(self.n):

            # for numerical feasability, normalize such that pm[j,j] = 0.
            # this is mathematically not necessary, but numerically important,
            # because the range of float64 would be exhausted pretty quickly.
            probmatrix[:, j, :] = probmatrix[:, j, :] - probmatrix[j, j, :]

            # normalize each row "properly",
            # such that the probabilites sum up to 1.
            probmatrix[:, j, :] = probmatrix[:, j, :] - \
                np.log(np.sum(np.e**probmatrix[:, j, :], axis=0))

        # p(H|x)
        self.log_p_H_x = np.copy(probmatrix)

    def calculate_utility(self, utility=None):

        if utility == "information":
            return self.entropyGain

        elif utility == "confidence":
            return self.confidence

        elif utility == "wrong":
            return self.wrong

    @cached_property
    def entropyGain(self):
        log_p_H_x = self.log_p_H_x
        log_p_x_H = self.log_p_x_H

        # reshape
        new_log_p_H_x = np.zeros((self.n, self.n * self.samples))
        new_log_p_x_H = np.zeros((self.n, self.n * self.samples))

        for i in range(self.n):
            new_log_p_H_x[i, :] = log_p_H_x[i, :, :].flatten()
            new_log_p_x_H[i, :] = log_p_x_H[i, :, :].flatten()

        log_p_H_x = new_log_p_H_x
        log_p_x_H = new_log_p_x_H

        # prior entropy
        # E_before = - np.log(1 / self.n)

        # normalize and sum
        log_p_x_H -= np.mean(log_p_x_H)    # normalize for better numerics
        p_x = np.sum(np.e**log_p_x_H, axis=0)
        p_x = p_x / np.sum(p_x)  # normalize so that it sums to 1

        # posterior utility
        U_x = - log_p_H_x * np.exp(log_p_H_x)
        U_x = np.sum(U_x, axis=0)  # sum over hpyothesis
        U_x = U_x / np.log(2)   # convert to bits

        U = np.mean(U_x)  # expectation value of U_x

        # print('entropy after:', U_d)
        E_after = U

        # utility_gain = E_before - E_after

        # calculate error
        err = np.std(U_x) / np.sqrt(self.samples)

        for spectrum in self.spectra:
            del spectrum.sample_spectrum

        return E_after, err

    @cached_property
    def wrong(self):
        """Probability of infering a hypothesis that is not equal to the ground truth."""
        log_p_H_x = self.log_p_H_x
        p_H_x = np.exp(log_p_H_x)

        successes = []

        for i in range(self.n):

            conf = p_H_x[:, i, :]
            maxidx = np.argmax(conf, axis=0)
            # prob is True where inference found the ground truth
            prob = maxidx == i
            successes.append(prob)

        prob_of_success = np.sum(successes) / (self.samples * self.n)

        for spectrum in self.spectra:
            del spectrum.sample_spectrum

        # TODO: Error
        err = 0

        return 1 - prob_of_success, err

    @cached_property
    def confidence(self):
        """Likelihood of being wrong"""
        log_p_H_x = self.log_p_H_x

        # reshape
        new_log_p_H_x = np.zeros((self.n, self.n * self.samples))

        for i in range(self.n):
            new_log_p_H_x[i, :] = log_p_H_x[i, :, :].flatten()

        p_H_x = np.exp(new_log_p_H_x)

        # get diagonal elements
        conf = np.amax(p_H_x, axis=0)

        err = np.std(conf) / np.sqrt(self.samples)

        # mean out the different samples
        conf = np.mean(conf)

        for spectrum in self.spectra:
            del spectrum.sample_spectrum

        return 1 - conf, err