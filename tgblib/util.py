#!/usr/bin/python

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import math
import itertools
from itertools import cycle
from itertools import combinations

import naima
from naima.models import (
    ExponentialCutoffPowerLaw,
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton
)

degToRad = u.deg.to(u.rad)


def my_energy_labels(ax):
    energy_units = {-6: 'μeV', -3: 'meV', 0: 'eV',
                    3: 'keV', 6: 'MeV', 9: 'GeV',
                    12: 'TeV', 15: 'PeV', 18: 'EeV'}
    xticks, xticklabels = list(), list()
    for i in np.linspace(-6, 18, 25):
        xticks.append(10**i)
        left = i % 3
        label = energy_units[i] if left == 0 else ''
        xticklabels.append(label)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def ChiSq(m, f, s):
    return pow((m-f)/s, 2)


vecChiSq = np.vectorize(ChiSq)


def set_my_fonts(mode='text'):
    if mode == 'text':
        plt.rc('font', family='serif', size=15)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('text', usetex=True)
    elif mode == 'talk':
        plt.rc('font', family='serif', size=20)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.rc('text', usetex=True)
    elif mode == 'large':
        plt.rc('font', family='serif', size=25)
        plt.rc('xtick', labelsize=25)
        plt.rc('ytick', labelsize=25)
        plt.rc('text', usetex=True)
