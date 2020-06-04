#!/usr/bin/python3

import matplotlib.pyplot as plt
import logging

import astropy.units as u

from tgblib import util
from tgblib.data import get_data
from tgblib.parameters import MJD_MEAN, NO_OF_PERIODS
from tgblib.spectrum_fit import SpectrumFit


if __name__ == '__main__':

    util.set_my_fonts(mode='talk')
    show = True
    label = 'std'

    for iper in range(NO_OF_PERIODS):
        logging.info('Plotting data - Period {}'.format(iper))

        nuStarEnergy, nuStarFlux, nuStarFluxErr = get_data(iper, onlyNuSTAR=True)
        vtsEnergy, vtsFlux, vtsFluxErr = get_data(iper, onlyVTS=True)

        ##########
        # NuSTAR
        if len(nuStarEnergy) == 0:
            logging.info('NuSTAR data is empty - skipping')
        else:
            plt.figure(figsize=(8, 6), tight_layout=True)
            ax = plt.gca()
            ax.set_title('Period {}'.format(iper))
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

            plt.savefig(
                'figures/DataNuStar_' + str(iper) + '_' + label + '.png',
                format='png',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/DataNuStar_' + str(iper) + '_' + label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )

        #########
        # VERITAS
        if len(vtsEnergy) == 0:
            logging.info('VTS data is empty - skipping')
        else:
            plt.figure(figsize=(8, 6), tight_layout=True)
            ax = plt.gca()
            ax.set_title('Period {}'.format(iper))
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
            ax.set_xlabel(r'$E\;[\mathrm{keV}]$')

            sf = SpectrumFit(energy=vtsEnergy, spec=vtsFlux, specErr=vtsFluxErr)

            sf.fit_power_law(Emin=util.get_emin_fit(vtsEnergy), Emax=util.get_emax_fit(vtsEnergy))
            sf.plot_data(color='k', marker='o', linestyle='None')
            sf.plot_fit(color='k', linestyle='-', linewidth=1.5)
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

            plt.savefig(
                'figures/DataVTS_' + str(iper) + '_' + label + '.png',
                format='png',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/DataVTS_' + str(iper) + '_' + label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )

    if show:
        plt.show()
