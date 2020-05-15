#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import itertools
from itertools import combinations
from numpy import arccos
from numpy.linalg import norm
from scipy import optimize

import astropy.constants as const
import astropy.units as u
from iminuit import Minuit

from tgblib import util
from tgblib import orbit
from tgblib import data
from tgblib.spectrum_fit import SpectrumFit


def computeEmin(en):
    return 10**(math.log10(en[0]) - (math.log10(en[1]) - math.log10(en[0])) / 2)


def computeEmax(en):
    return 10**(math.log10(en[-1]) + (math.log10(en[-1]) - math.log10(en[-2])) / 2)


def test_plot_data():
    '''
    Energies in keV
    Integrated fluxes given in erg cm-2 s-1
    '''

    for iper in range(2):

        data_en, data_fl, data_fl_er = data.get_data(iper, '5sig')

        nuStarEnergy = [e for e in data_en if e < 1e6]
        nuStarFlux = [f for (f, e) in zip(data_fl, data_en) if e < 1e6]
        nuStarFluxErr = [f for (f, e) in zip(data_fl_er, data_en) if e < 1e6]

        vtsEnergy = [e for e in data_en if e > 1e3]
        vtsFlux = [f for (f, e) in zip(data_fl, data_en) if e > 1e3]
        vtsFluxErr = [f for (f, e) in zip(data_fl_er, data_en) if e > 1e3]

        ##########
        # NuSTAR
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
        ax.set_xlabel(r'$E\;[\mathrm{keV}]$')
        ax.tick_params(which='minor', length=5)
        ax.tick_params(which='major', length=9)

        sf = SpectrumFit(energy=nuStarEnergy, spec=nuStarFlux, specErr=nuStarFluxErr)

        sf.fit_power_law(Emin=3, Emax=30)
        sf.plot_data(color='k', marker='o', linestyle='None')
        sf.plot_fit(color='k', linestyle='-', linewidth=1.5)
        # sf.plot_fit_unc(color='g', linestyle='None', alpha=0.15, hatch='/')
        sf.plot_fit_unc(nocor=True, color='k', linestyle='None', alpha=0.2)

        ylim = ax.get_ylim()
        ax.set_ylim(2.5e-13, 40e-13)
        ax.set_xlim(2.8, 35)

        myTicks = [3, 10, 30]
        ax.set_xticks(myTicks)
        ax.set_xticklabels(myTicks)

        ax.set_yticks([3e-13, 1e-12])
        ax.set_yticklabels([r'$3\;10^{-13}$', r'$10^{-12}$'])

        print('Flux = {}+/-{}'.format(sf.flux, sf.fluxErr))
        print('Slope = {}+/-{}'.format(sf.gamma, sf.gammaErr))

        plt.show()

        #########
        # VERITAS
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
        ax.set_xlabel(r'$E\;[\mathrm{keV}]$')

        sf = SpectrumFit(energy=vtsEnergy, spec=vtsFlux, specErr=vtsFluxErr)

        sf.fit_power_law(Emin=computeEmin(vtsEnergy), Emax=computeEmax(vtsEnergy))
        sf.plot_data(color='k', marker='o', linestyle='None')
        sf.plot_fit(color='k', linestyle='-', linewidth=1.5)
        # sf.plot_fit_unc(color='g', linestyle='None', alpha=0.15, hatch='/')
        sf.plot_fit_unc(nocor=True, color='k', linestyle='None', alpha=0.2)

        ylim = ax.get_ylim()
        ax.set_ylim(1.7e-14, 9e-12)
        ax.set_xlim(1.8e8, 3.5e9)

        myTicks = [3e8, 1e9, 3e9]
        myLabels = [r'$3\;10^{8}$', r'$10^{9}$', r'$3\;10^{9}$']
        ax.set_xticks(myTicks)
        ax.set_xticklabels(myLabels)

        print('Flux = {}+/-{}'.format(sf.flux, sf.fluxErr))
        print('Slope = {}+/-{}'.format(sf.gamma, sf.gammaErr))

        # Normalization - Gernot's question Jan2020
        N = sf.get_norm(e=1e9)  # in erg cm-2 s-1
        print('Norm (1 TeV) = {}'.format(N))

        # erg->TeV convertion
        convEn = u.erg.to(u.TeV)

        # f = dN/dE
        f = N / (1 / convEn)**2  # in erg-1 cm-2 s-1
        print('dN/dE (1 TeV) = {} [erg-1 cm-2 s-1]'.format(f))
        print('dN/dE (1 TeV) = {} [TeV-1 cm-2 s-1]'.format(f / convEn))

        plt.show()
