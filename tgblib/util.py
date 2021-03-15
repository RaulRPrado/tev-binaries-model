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


def get_emin_fit(en):
    return 10**(math.log10(en[0]) - (math.log10(en[1]) - math.log10(en[0])) / 2)


def get_emax_fit(en):
    return 10**(math.log10(en[-1]) + (math.log10(en[-1]) - math.log10(en[-2])) / 2)


def fix_naima_bug(energy, model, bin_size=0.1):
    ''' Fixing Naima bug at high energy buy smoothing the model out '''
    lge_bins = np.linspace(-1, 14, (14 + 1) * 10 + 1)

    new_energy, new_model = list(), list()
    for i in range(len(lge_bins) - 1):
        en = 10**((lge_bins[i] + lge_bins[i + 1]) * 0.5)
        collected_model = list()
        for (e, m) in zip(energy, model):
            if math.log10(e.value) > lge_bins[i + 1] or math.log10(e.value) < lge_bins[i]:
                continue
            collected_model.append(m)
        if len(collected_model) == 0:
            continue
        new_energy.append(en)
        new_model.append(np.mean(collected_model))

    return new_energy, new_model


def smooth_break(energy, model):
    new_model = list()
    for ii, en, mo in zip(range(len(energy)), energy, model):
        if en < 1.5e7 or en > 6e8:
            new_model.append(mo)
            continue

        nn = 3
        mm = sum(model[ii - nn:ii + nn]) / (2 * nn)

        new_model.append(mm)

    return new_model
