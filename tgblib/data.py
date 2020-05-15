#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from scipy import optimize

import astropy.units as u
import naima
from iminuit import Minuit
from astropy.table import QTable
from astropy.io import ascii
from naima.models import (
    ExponentialCutoffPowerLaw,
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton
)

from tgblib import util


def get_fermi_spec():
    # FERMI DATA
    energy_spec_fermi = [330286.4, 742891.2, 1641547.5]
    spec_fermi = [0.00000000000365453783, 0.00000000000244200875, 0.00000000000167663054]
    spec_fermi_hi = [0.00000000000498570704, 0.00000000000324273583, 0.00000000000212360791]
    spec_fermi_lo = [0.00000000000237221170, 0.00000000000170736167, 0.00000000000125412415]

    spec_er_hi = [h - c for (c, h) in zip(spec_fermi, spec_fermi_hi)]
    spec_er_lo = [c - l for (c, l) in zip(spec_fermi, spec_fermi_lo)]
    spec_er = [(h + l) / 2 for (h, l) in zip(spec_er_hi, spec_er_lo)]

    return energy_spec_fermi, spec_fermi, spec_er


def get_fermi_upper_limits():
    energy_lim_fermi = [147074.7, 3693286.9, 8398660.9, 18716341.4,
                        42967877.0, 94669591.3, 213894710.8]
    lim_fermi = [0.00000000000820099955, 0.00000000000120672819,
                 0.00000000000136532261, 0.00000000000083570012,
                 0.00000000000109690299, 0.00000000000355920154,
                 0.00000000000759721985]

    lim_er_fermi = [p - pow(10, math.log10(p)-0.1) for p in lim_fermi]

    return energy_lim_fermi, lim_fermi, lim_er_fermi


def get_data(period=0, which='3sig'):
    if period != 0 and period != 1:
        print('get_data:wrong period')
        return None, None, None
    data = ascii.read('data/HESS_J0632_' + str(period) + '_' + which + '.ecsv', format='basic')
    energy = list(data['energy'])                                    # keV
    flux = [f * u.keV.to(u.erg) for f in data['flux']]               # ergs / cm2 / s
    flux_lo = [f * u.keV.to(u.erg) for f in data['flux_error_lo']]  # ergs / cm2 / s
    flux_hi = [f * u.keV.to(u.erg) for f in data['flux_error_hi']]  # ergs / cm2 / s
    # flux_err = flux_hi  # ergs / cm2 / s
    flux_err = list()
    for (h, l, e) in zip(flux_hi, flux_lo, energy):
        if e < 1e3:
            flux_err.append(l)
        else:
            flux_err.append((l + h) / 2)
    return energy, flux, flux_err


def comp_phase(mjd_0, T, mjd):
    ph = (mjd - mjd_0) / T
    ph = ph - int(ph)
    if ph < 0:
        ph += 1
    return ph


def flux_mean(fl, fl_err):
    w_sum, fl_sum = 0, 0
    for (f, e) in zip(fl, fl_err):
        weight = 1 / e**2
        w_sum += weight
        fl_sum += weight * f
    return fl_sum / w_sum, 1 / math.sqrt(w_sum)


def ratio_with_error(y, yerr, x, xerr):
    r = y / x
    e = r * math.sqrt((yerr / y)**2 + (xerr / x)**2)
    return r, e


def get_lc_data(mjd_0=54857.5, no_bin=True, T=315, n_bins=5, which_phase=None):
    data_vts = ascii.read('data/VERITAS-LC-20181214.ecsv', format='basic')
    data_xray = np.loadtxt('data/xray-20180426.dat', comments='#', usecols=(1, 2, 3, 4),
                           skiprows=2, unpack=True)

    if which_phase is None:
        def which_phase(x):
            return True

    mjd_vts = list((data_vts['time_min'] + data_vts['time_max']) / 2)
    phase_vts = [comp_phase(mjd_0=mjd_0, T=T, mjd=t) for t in mjd_vts]
    fl_vts = list(data_vts['flux'])
    fl_err_vts = list(data_vts['flux_err'])

    phase_xray = [comp_phase(mjd_0=mjd_0, T=T, mjd=t) for t in data_xray[0]]
    fl_xray = [10**l for l in data_xray[1]]
    fl_err_xray = [(10**u-10**l) / 2 for (u, l) in zip(data_xray[3], data_xray[2])]

    if no_bin:
        return phase_xray, fl_xray, fl_err_xray, phase_vts, fl_vts, fl_err_vts

    bins = np.linspace(0, 1, n_bins + 1)
    phase_vts_bin, fl_vts_bin, fl_err_vts_bin = list(), list(), list()
    phase_xray_bin, fl_xray_bin, fl_err_xray_bin = list(), list(), list()
    for i in range(len(bins) - 1):
        phase = (bins[i] + bins[i+1]) / 2
        if not which_phase(phase):
            continue
        # VTS
        fl_sum, w_sum = 0, 0
        for (ph, fl, fl_err) in zip(phase_vts, fl_vts, fl_err_vts):
            if ph > bins[i] and ph < bins[i + 1]:
                weight = 1 / fl_err**2
                w_sum += weight
                fl_sum += weight * fl

        if w_sum > 0:
            phase_vts_bin.append(phase)
            fl_vts_bin.append(fl_sum / w_sum)
            fl_err_vts_bin.append(1 / math.sqrt(w_sum))

        # xrays
        fl_sum, w_sum = 0, 0
        for (ph, fl, fl_err) in zip(phase_xray, fl_xray, fl_err_xray):
            if ph > bins[i] and ph < bins[i + 1]:
                weight = 1 / fl_err**2
                w_sum += weight
                fl_sum += weight * fl

        if w_sum > 0:
            phase_xray_bin.append(phase)
            fl_xray_bin.append(fl_sum / w_sum)
            fl_err_xray_bin.append(1 / math.sqrt(w_sum))

    return phase_xray_bin, fl_xray_bin, fl_err_xray_bin, phase_vts_bin, fl_vts_bin, fl_err_vts_bin


def get_lc_ratio_data(mjd_0=54857.5, T=315, which_phase=None, max_err=1e10):
    data_vts = ascii.read('data/VERITAS-LC-20181214.ecsv', format='basic')
    data_xray = np.loadtxt('data/xray-20180426.dat', comments='#', usecols=(1, 2, 3, 4),
                           skiprows=2, unpack=True)

    if which_phase is None:
        def which_phase(x):
            return True

    mjd_vts = list((data_vts['time_min'] + data_vts['time_max']) / 2)
    mjd_min_vts = list(data_vts['time_min'])
    mjd_max_vts = list(data_vts['time_max'])
    fl_vts = list(data_vts['flux'])
    fl_err_vts = list(data_vts['flux_err'])

    mjd_xray = list(data_xray[0])
    fl_xray = [10**l for l in data_xray[1]]
    fl_err_xray = [(10**u-10**l) / 2 for (u, l) in zip(data_xray[3], data_xray[2])]

    mjd, phase = list(), list()
    flux_vts, flux_err_vts = list(), list()
    flux_xray, flux_err_xray = list(), list()
    ratio, ratio_err = list(), list()

    for ig in range(len(mjd_vts)):
        if fl_vts[ig] < 0:
            continue
        ph_vts = comp_phase(mjd_0=mjd_0, T=T, mjd=mjd_vts[ig])
        if not which_phase(ph_vts):
            continue

        fl_xray_sel, fl_err_xray_sel = list(), list()

        for ix in range(len(mjd_xray)):
            if mjd_xray[ix] > mjd_min_vts[ig] and mjd_xray[ix] < mjd_max_vts[ig]:
                fl_xray_sel.append(fl_xray[ix])
                fl_err_xray_sel.append(fl_err_xray[ix])

        if len(fl_xray_sel) == 0:
            continue
        fl, fl_err = flux_mean(fl=fl_xray_sel, fl_err=fl_err_xray_sel)
        r, r_err = ratio_with_error(y=fl, yerr=fl_err, x=fl_vts[ig], xerr=fl_err_vts[ig])
        if r_err / r > max_err:
            continue

        mjd.append(mjd_vts[ig])
        phase.append(ph_vts)
        flux_xray.append(fl)
        flux_err_xray.append(fl_err)
        flux_vts.append(fl_vts[ig])
        flux_err_vts.append(fl_err_vts[ig])
        ratio.append(r)
        ratio_err.append(r_err)

    return mjd, phase, flux_xray, flux_err_xray, flux_vts, flux_err_vts, ratio, ratio_err


if __name__ == '__main__':
    util.set_my_fonts(mode='talk')

    def which_phase(x):
        return True

    mjd, phase, fl_xray, fl_err_xray, fl_vts, fl_err_vts, ratio, ratio_err =\
        get_lc_ratio_data(T=315, which_phase=which_phase, max_err=0.9)

    plt.figure(figsize=(24, 12), tight_layout=True)
    plt.subplot(2, 3, 1)
    ax = plt.gca()
    ax.set_xlabel(r'$\phi$')

    ax.errorbar(phase, fl_xray, yerr=fl_err_xray,
                linestyle='None', color='k', marker='o')

    plt.subplot(2, 3, 2)
    ax = plt.gca()
    ax.set_xlabel(r'$\phi$')

    ax.errorbar(phase, fl_vts, yerr=fl_err_vts,
                linestyle='None', color='k', marker='o')

    plt.subplot(2, 3, 3)
    ax = plt.gca()
    ax.set_xlabel(r'$\phi$')

    ax.errorbar(phase, ratio, yerr=ratio_err,
                linestyle='None', color='k', marker='o')

    ##################################

    phase_xray, fl_xray, fl_err_xray, phase_vts, fl_vts, fl_err_vts =\
        get_lc_data(no_bin=False, n_bins=10, T=315, which_phase=which_phase)

    ratio, ratio_err = list(), list()
    for i in range(len(phase_xray)):
        r, err = ratio_with_error(y=fl_xray[i], yerr=fl_err_xray[i], x=fl_vts[i], xerr=fl_err_vts[i])
        ratio.append(r)
        ratio_err.append(err)

    plt.subplot(2, 3, 4)
    ax = plt.gca()
    ax.set_xlabel(r'$\phi$')

    ax.errorbar(phase_xray, fl_xray, yerr=fl_err_xray,
                linestyle='None', color='k', marker='o')

    plt.subplot(2, 3, 5)
    ax = plt.gca()
    ax.set_xlabel(r'$\phi$')

    ax.errorbar(phase_vts, fl_vts, yerr=fl_err_vts,
                linestyle='None', color='k', marker='o')

    plt.subplot(2, 3, 6)
    ax = plt.gca()
    ax.set_xlabel(r'$\phi$')

    ax.errorbar(phase_vts, ratio, yerr=ratio_err,
                linestyle='None', color='k', marker='o')

    plt.show()

