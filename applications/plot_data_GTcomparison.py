#!/usr/bin/python3

import matplotlib.pyplot as plt
import logging
import math

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
        0: 'with GT corrections',
        1: 'without GT corrections',
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
    fluxes = dict()
    for iper in [0, 1]:
        fluxes[iper] = dict()
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Nov. 2017' if iper == 0 else 'Dec. 2017')
        ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
        ax.set_xlabel(r'$E\;[\mathrm{keV}]$')
        ax.tick_params(which='minor', length=MINOR_TICK)
        ax.tick_params(which='major', length=MAJOR_TICK)

        for nn, gt in enumerate([True, False]):
            vtsEnergy, vtsFlux, vtsFluxErr = get_data(iper, onlyVTS=True, GT=gt)
            vtsEnergyUL, vtsFluxUL = get_data_ul(iper, GT=gt)
            ax.errorbar(
                [e * (1 + 0.02 * nn) for e in vtsEnergy],
                vtsFlux,
                yerr=vtsFluxErr,
                color=COLORS[nn],
                linestyle='none',
                label=VTS_TITLE[nn],
                marker=MARKERS[nn]
            )
            if len(vtsEnergyUL) > 0:
                vtsFluxErrUL = [p - pow(10, math.log10(p) - 0.1) for p in vtsFluxUL]
                ax.errorbar(
                    vtsEnergyUL,
                    vtsFluxUL,
                    yerr=vtsFluxErrUL,
                    uplims=True,
                    color=COLORS[nn],
                    linestyle='none',
                    marker=MARKERS[nn]
                )
            fluxes[iper][nn] = vtsFlux

        ax.set_ylim(0.8e-13, 5e-12)
        ax.set_xlim(1e8, 2e10)

        myTicks = [1e8, 1e9, 1e10]
        myLabels = [r'$10^{8}$', r'$10^{9}$', r'$10^{10}$']
        ax.set_xticks(myTicks)
        ax.set_xticklabels(myLabels)

        ax.legend(loc='best', frameon=False)

        figName = 'figures/DataVTS_GTcomparison_{}'.format(iper)
        plt.savefig(figName + '.png', format='png', bbox_inches='tight')
        plt.savefig(figName + '.pdf', format='pdf', bbox_inches='tight')

    # Calculating ratios
    for iper in [0, 1]:
        nn = 3 if iper == 0 else 2
        for ii in range(nn):
            print(ii)
            a = fluxes[iper][0][ii]
            b = fluxes[iper][1][ii + 1]
            ratio = (a - b) / (a + b) / 2
            print(ratio)
