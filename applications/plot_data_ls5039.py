#!/usr/bin/python3

import matplotlib.pyplot as plt
import logging
import math

import astropy.units as u

from tgblib import util
from tgblib.data import get_data_ls5039
from tgblib.parameters import MJD_MEAN, NO_OF_PERIODS, MONTH_LABEL
from tgblib.spectrum_fit import SpectrumFit

logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':

    util.set_my_fonts(mode='talk')
    show = True

    for obs in ['SUZAKU', 'HESS']:
        print(obs)

        phase, flux, fluxErr, gamma, gammaErr = get_data_ls5039(obs)

        # FLUX
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_title(obs)
        # ax.set_yscale('log')
        if obs == 'HESS':
            ax.set_ylabel(r'dN/dE ($10^{-12}\;$TeV$^{-1}\;$cm$^{-2}\;$s$^{-1}$) at 1 TeV ')
        else:
            ax.set_ylabel(r'F$_{1-10\;\mathrm{keV}}$ ($10^{-12}\;$erg$\;$cm$^{-2}\;$s$^{-1}$)')

        ax.set_xlabel(r'orbital phase')

        ax.errorbar(phase, flux, yerr=fluxErr, color='k', linestyle='none', marker='o')

        ax.set_xlim(-0.01, 1.01)

        if show:
            plt.show()
        else:
            for f in ['pdf', 'png']:
                plt.savefig(
                    'figures/LS5039/Data' + obs + '_flux.' + f,
                    format=f,
                    bbox_inches='tight'
                )

        # GAMMA
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_title(obs)
        # ax.set_yscale('log')
        ax.set_ylabel(r'$\Gamma$')
        ax.set_xlabel(r'orbital phase')
        # ax.tick_params(which='minor', length=5)
        # ax.tick_params(which='major', length=9)

        ax.errorbar(phase, gamma, yerr=gammaErr, color='k', linestyle='none', marker='o')

        if show:
            plt.show()
        else:
            for f in ['pdf', 'png']:
                plt.savefig(
                    'figures/LS5039/Data' + obs + '_gamma.' + f,
                    format=f,
                    bbox_inches='tight'
                )
