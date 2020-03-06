#!/usr/bin/env python3

from spectres import spectres
import numpy as np
import matplotlib.pyplot as plt

times = ['modern', '0.8Ga', '2.0Ga', '3.9Ga']

for time in times:
    flux = np.loadtxt('spectra/cloudy/' + time + '.dat')[:,1]
    wl = np.loadtxt('spectra/cloudy/' + time + '.dat')[:,0]

    flux = np.flip(flux)
    wl = np.flip(wl)

    newwl = np.linspace(np.amin(wl)+0.1, np.amax(wl)-0.1, 500)
    newflux = spectres(newwl, wl, flux)
    plt.plot(newwl, newflux, label=time, linewidth=1)


plt.yscale('log')
#plt.xscale('log')
plt.xlabel('wavelength [um]')
plt.ylabel('flux TOA [W/m²/um]')
plt.legend()
plt.show()
