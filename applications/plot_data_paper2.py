#!/usr/bin/python3

import matplotlib.pyplot as plt
import logging
import math

from astropy import units as u

from tgblib import util
from tgblib.data import get_data, get_data_ul

logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':

    util.set_my_fonts(mode='talk')
    show = False
    label = 'std'

    NU_TITLE = {
        0: 'Nu1a',
        1: 'Nu1b',
        2: 'Nu2a',
        3: 'none',
        4: 'Nu2b'
    }
    VTS_TITLE = {
        0: 'Ve1a',
        1: 'Ve1b',
        2: 'Ve2a',
        3: 'Ve2b',
        4: 'Ve2c'
    }

    MARKERS = {
        0: 'o',
        1: 's',
        2: 'o',
        3: 's',
        4: '*'
    }

    COLORS = {
        0: 'k',
        1: 'r',
        2: 'k',
        3: 'r',
        4: 'b'
    }

    MINOR_TICK = 7.5
    MAJOR_TICK = 12

    # 2017
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
    ax.set_xlabel(r'$E\;[\mathrm{keV}]$')
    ax.tick_params(which='minor', length=MINOR_TICK)
    ax.tick_params(which='major', length=MAJOR_TICK)

    for nn, iper in enumerate([0, 1]):
        vtsEnergy, vtsFlux, vtsFluxErr = get_data(iper, onlyVTS=True, GT=True)
        vtsEnergyUL, vtsFluxUL = get_data_ul(iper, GT=True)
        ax.errorbar(
            [e * (1 + 0.02 * nn) for e in vtsEnergy],
            vtsFlux,
            yerr=vtsFluxErr,
            color=COLORS[iper],
            linestyle='none',
            label=VTS_TITLE[iper],
            marker=MARKERS[iper]
        )
        if len(vtsEnergyUL) > 0:
            vtsFluxErrUL = [p - pow(10, math.log10(p) - 0.1) for p in vtsFluxUL]
            ax.errorbar(
                vtsEnergyUL,
                vtsFluxUL,
                yerr=vtsFluxErrUL,
                uplims=True,
                color=COLORS[iper],
                linestyle='none',
                marker=MARKERS[iper]
            )

    ax.set_ylim(0.8e-13, 5e-12)
    ax.set_xlim(3e8, 2e10)

    myTicks = [1e9, 1e10]
    myLabels = [r'$10^{9}$', r'$10^{10}$']
    ax.set_xticks(myTicks)
    ax.set_xticklabels(myLabels)

    ax.legend(loc='best', frameon=False)

    plt.savefig(
        'figures/DataVTS_2017.png',
        format='png',
        bbox_inches='tight'
    )
    plt.savefig(
        'figures/DataVTS_2017.pdf',
        format='pdf',
        bbox_inches='tight'
    )

    # 2019
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
    ax.set_xlabel(r'$E\;[\mathrm{TeV}]$')
    ax.tick_params(which='minor', length=MINOR_TICK)
    ax.tick_params(which='major', length=MAJOR_TICK)

    keV_to_TeV = u.keV.to(u.TeV)

    for nn, iper in enumerate([2, 3, 4]):
        vtsEnergy, vtsFlux, vtsFluxErr = get_data(iper, onlyVTS=True)
        vtsEnergyUL, vtsFluxUL = get_data_ul(iper)
        ax.errorbar(
            [e * (1 + 0.02 * nn) * keV_to_TeV for e in vtsEnergy],
            vtsFlux,
            yerr=vtsFluxErr,
            color=COLORS[iper],
            linestyle='none',
            label=VTS_TITLE[iper],
            marker=MARKERS[iper]
        )
        if len(vtsEnergyUL) > 0:
            vtsFluxErrUL = [p - pow(10, math.log10(p) - 0.1) for p in vtsFluxUL]
            ax.errorbar(
                [e * keV_to_TeV for e in vtsEnergyUL],
                vtsFluxUL,
                yerr=vtsFluxErrUL,
                uplims=True,
                color=COLORS[iper],
                linestyle='none',
                marker=MARKERS[iper]
            )

    ax.set_ylim(0.8e-13, 5e-12)
    ax.set_xlim(3e-1, 2e1)

    myTicks = [1e0, 1e1]
    myLabels = [r'$10^{0}$', r'$10^{1}$']
    ax.set_xticks(myTicks)
    ax.set_xticklabels(myLabels)

    ax.legend(loc='best', frameon=False)

    plt.savefig(
        'figures/DataVTS_2019.png',
        format='png',
        bbox_inches='tight'
    )
    plt.savefig(
        'figures/DataVTS_2019.pdf',
        format='pdf',
        bbox_inches='tight'
    )
