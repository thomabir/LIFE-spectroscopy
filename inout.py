import numpy as np
import pandas as pd
from functools import lru_cache

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