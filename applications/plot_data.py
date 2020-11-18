#!/usr/bin/python3

import matplotlib.pyplot as plt
import logging
import math

import astropy.units as u

from tgblib import util
from tgblib.data import get_data, get_data_ul
from tgblib.parameters import MJD_MEAN, NO_OF_PERIODS, MONTH_LABEL
from tgblib.spectrum_fit import SpectrumFit

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

    for iper in range(NO_OF_PERIODS):
        logging.info('Plotting data - Period {}'.format(iper))

        nuStarEnergy, nuStarFlux, nuStarFluxErr = get_data(iper, onlyNuSTAR=True)
        vtsEnergy, vtsFlux, vtsFluxErr = get_data(iper, onlyVTS=True)
        vtsEnergyUL, vtsFluxUL = get_data_ul(iper)

        ##########
        # NuSTAR
        if len(nuStarEnergy) == 0:
            logging.info('NuSTAR data is empty - skipping')
        else:
            logging.info('NuSTAR Period: {}'.format(iper))

            plt.figure(figsize=(8, 6), tight_layout=True)
            ax = plt.gca()
            ax.set_title(NU_TITLE[iper])
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
            ax.set_ylim(2.5e-13, 6e-12)
            ax.set_xlim(2.8, 35)

            myTicks = [3, 10, 30]
            ax.set_xticks(myTicks)
            ax.set_xticklabels(myTicks)

            ax.set_yticks([3e-13, 1e-12])
            ax.set_yticklabels([r'$3\;10^{-13}$', r'$10^{-12}$'])

            flux_box = sf.flux / 1e-12
            flux_err_box = sf.fluxErr / 1e-12

            flux_text = (
                r'$F_\mathrm{3-30 keV}$ = ' + '({:.2f}'.format(flux_box) + r'$\pm$'
                + '{:.2f}'.format(flux_err_box) + r') $10^{12}$ erg s$^{-1}$'
            )
            slope_text = (
                r'$\Gamma$ = ' + '{:.2f}'.format(sf.gamma) + r'$\pm$'
                + '{:.2f}'.format(sf.gammaErr)
            )

            # ax.text(
            #     0.07,
            #     0.90,
            #     flux_text,
            #     transform=ax.transAxes
            # )
            # ax.text(
            #     0.07,
            #     0.82,
            #     slope_text,
            #     transform=ax.transAxes
            # )

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
        if len(vtsEnergy) == 0 and len(vtsEnergyUL) == 0:
            logging.info('VTS data is empty - skipping')
        else:
            logging.info('VTS Period: {}'.format(iper))

            plt.figure(figsize=(8, 6), tight_layout=True)
            ax = plt.gca()
            ax.set_title(VTS_TITLE[iper])
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
            ax.set_xlabel(r'$E\;[\mathrm{keV}]$')

            if len(vtsEnergy) > 0:
                sf = SpectrumFit(energy=vtsEnergy, spec=vtsFlux, specErr=vtsFluxErr)

                # sf.fit_power_law(Emin=util.get_emin_fit(vtsEnergy), Emax=util.get_emax_fit(vtsEnergy))
                sf.plot_data(color='k', marker='o', linestyle='None')
                if len(vtsEnergy) > 1:
                    sf.fit_power_law(Emin=0.2e9, Emax=3e9)
                    sf.plot_fit(color='k', linestyle='-', linewidth=1.5)
                    sf.plot_fit_unc(nocor=True, color='k', linestyle='None', alpha=0.2)

                    flux_box = sf.flux / 1e-12
                    flux_err_box = sf.fluxErr / 1e-12

                    flux_text = (
                        r'$F_\mathrm{0.2-3 TeV}$ = ' + '({:.2f}'.format(flux_box) + r'$\pm$'
                        + '{:.2f}'.format(flux_err_box) + r') $10^{12}$ erg s$^{-1}$'
                    )
                    slope_text = (
                        r'$\Gamma$ = ' + '{:.2f}'.format(sf.gamma) + r'$\pm$'
                        + '{:.2f}'.format(sf.gammaErr)
                    )

                    # ax.text(
                    #     0.07,
                    #     0.12,
                    #     flux_text,
                    #     transform=ax.transAxes
                    # )
                    # ax.text(
                    #     0.07,
                    #     0.05,
                    #     slope_text,
                    #     transform=ax.transAxes
                    # )

                    # Normalization - Gernot's question Jan2020
                    N = sf.get_norm(e=1e9)  # in erg cm-2 s-1
                    print('Norm (1 TeV) = {}'.format(N))

                    # erg->TeV convertion
                    convEn = u.erg.to(u.TeV)

                    # f = dN/dE
                    f = N / (1 / convEn)**2  # in erg-1 cm-2 s-1
                    print('dN/dE (1 TeV) = {} [erg-1 cm-2 s-1]'.format(f))
                    print('dN/dE (1 TeV) = {} [TeV-1 cm-2 s-1]'.format(f / convEn))

            # UPPER LIMITS
            if len(vtsEnergyUL) > 0:
                vtsFluxErrUL = [p - pow(10, math.log10(p)-0.1) for p in vtsFluxUL]
                ax.errorbar(
                    vtsEnergyUL,
                    vtsFluxUL,
                    yerr=vtsFluxErrUL,
                    uplims=True,
                    color='k',
                    linestyle='none'
                )

            ax.set_ylim(1.7e-14, 9e-12)
            ax.set_xlim(1e8, 1e10)

            myTicks = [1e8, 1e9, 1e10]
            myLabels = [r'$10^{8}$', r'$10^{9}$', r'$10^{10}$']
            ax.set_xticks(myTicks)
            ax.set_xticklabels(myLabels)

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
