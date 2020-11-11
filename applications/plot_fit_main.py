#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import math

import astropy.units as u
import naima
from iminuit import Minuit
from naima.models import (
    ExponentialCutoffPowerLaw,
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton
)

import tgblib.pulsar as psr
import tgblib.fit_results as fr
import tgblib.parameters as pars
from tgblib import util
from tgblib import data
from tgblib import orbit


if __name__ == '__main__':

    util.set_my_fonts(mode='talk')
    periods = [0, 1, 2, 4]
    # periods = [0, 1]
    plot_label = '_paper2'
    small_label = '_small'
    which_orbit = ['ca', 'mo']
    show = False
    band = True
    fast_sed = True
    do_solution = True
    do_sed = False
    do_sed_both = True
    do_mag = False
    do_density = False
    do_dist = False
    do_opt = False
    do_ebr = False      # no
    do_sig = False      # no
    do_mdot = False     # no

    label_ca = 'Orbit by Casares et al. 2012'
    label_mo = 'Orbit by Moritani et al. 2018'
    SigmaMax = 1e5
    EdotMin = 1e34

    minorTickSize = 4
    majorTickSize = 7

    ########
    # Orbits
    systems_ca = orbit.getCasaresSystem()
    systems_mo = orbit.getMoritaniSystem()

    mjd_pts = mjd_pts = [pars.MJD_MEAN[p] for p in periods]
    orbits_ca = orbit.SetOfOrbits(
        phase_step=0.0005,
        color='r',
        systems=systems_ca,
        mjd_pts=mjd_pts
    )
    orbits_mo = orbit.SetOfOrbits(
        phase_step=0.0005,
        color='b',
        systems=systems_mo,
        mjd_pts=mjd_pts
    )

    pts_ca = orbits_ca.get_pts()
    pts_mo = orbits_mo.get_pts()
    theta_ic_ca = pts_ca['theta_ic']
    dist_ca = pts_ca['distance']
    pos_ca = pts_ca['pos_3D']
    theta_ic_mo = pts_mo['theta_ic']
    dist_mo = pts_mo['distance']
    pos_mo = pts_mo['pos_3D']

    #############
    # Fit Results
    fr_ca = fr.FitResult(
        n_periods=len(periods),
        label='ca' + small_label,
        color='r',
        SigmaMax=SigmaMax,
        EdotMin=EdotMin
    )
    fr_mo = fr.FitResult(
        n_periods=len(periods),
        label='mo' + small_label,
        color='b',
        SigmaMax=SigmaMax,
        EdotMin=EdotMin
    )

    xlim, ylim = [8e33, 3e38], [1e-3, 1e-1]

    def plot_solution():
        # Solution - main
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$\sigma_0$')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=majorTickSize)

        fr_ca.plot_solution(
            band=band,
            line=True,
            ms=40,
            with_lines=True,
            no_2s=False,
            ls='-',
            line_ls='--',
            label=label_ca
        )

        fr_mo.plot_solution(
            band=band,
            line=True,
            ms=40,
            with_lines=True,
            no_2s=False,
            ls='-',
            line_ls=':',
            label=label_mo
        )

        ax.legend(loc='best', frameon=False)

        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitSolutionsBoth' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitSolutionsBoth' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

        # Solution - Casares
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$\sigma_0$')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=majorTickSize)

        fr_ca.plot_solution(
            band=band,
            line=True,
            ms=40,
            with_lines=True,
            no_2s=False,
            ls='-',
            line_ls='--',
            label=label_ca
        )

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

        ax.text(0.1, 0.08, label_ca, transform=ax.transAxes, horizontalalignment='left')

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitSolutionsCa' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitSolutionsCa' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

        # Solution - Moritani
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$\sigma_0$')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=majorTickSize)

        fr_mo.plot_solution(
            band=band,
            line=True,
            ms=40,
            with_lines=True,
            no_2s=False,
            ls='-',
            line_ls='--',
            label=label_mo
        )

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

        ax.text(0.1, 0.08, label_mo, transform=ax.transAxes, horizontalalignment='left')

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitSolutionsMo' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitSolutionsMo' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

    def plot_sed():
        ##############################
        # SED Casares
        # plt.figure(figsize=(len(periods)*8, 6), tight_layout=True)

        for orb in ['Ca', 'Mo']:

            plt.figure(figsize=(2 * 8, 2 * 6), tight_layout=True)
            for iper, per in enumerate(periods):
                plt.subplot(2, 2, iper + 1)
                fig = plt.gcf()
                ax = plt.gca()
                ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
                ax.set_xlabel(r'$E$ [keV]')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.tick_params(which='minor', length=minorTickSize)
                ax.tick_params(which='major', length=majorTickSize)

                if orb == 'Ca':
                    label_main = round((10**fr_ca.lgEdotMin) / (10**int(fr_ca.lgEdotMin)), 2)
                    label_pow = int(fr_ca.lgEdotMin)
                    label_sed = (
                        label_ca + '\n' + r'$L_\mathrm{sd}=$' +
                        str(label_main) + r'$\;10^{'+str(label_pow) + r'}$' +
                        r' ergs/s, $\sigma_0$=' + '{:.3f}'.format(10**fr_ca.lgSigmaMin)
                    )
                else:
                    label_main = round((10**fr_mo.lgEdotMin) / (10**int(fr_mo.lgEdotMin)), 2)
                    label_pow = int(fr_mo.lgEdotMin)
                    label_sed = (
                        label_mo + '\n' + r'$L_\mathrm{sd}=$' +
                        str(label_main) + r'$\;10^{'+str(label_pow) + r'}$' +
                        r' ergs/s, $\sigma_0$=' + '{:.3f}'.format(10**fr_mo.lgSigmaMin)
                    )

                if iper == 0:
                    ax.text(
                        0.05,
                        0.85,
                        label_sed,
                        transform=ax.transAxes,
                        horizontalalignment='left'
                    )

                if orb == 'Ca':
                    fr_ca.plot_sed(
                        iperiod=iper,
                        period=per,
                        theta_ic=theta_ic_ca[iper],
                        dist=dist_ca[iper],
                        pos=pos_ca[iper],
                        ls='-',
                        label=r'$E_\mathrm{min}=0.1$ TeV, $E_\mathrm{cut}=50$ TeV',
                        emin=0.10,
                        ecut=50,
                        fast=fast_sed,
                        best_solution=False,
                        Edot=1e36
                    )
                    # fr_ca.plot_sed(
                    #     iperiod=iper,
                    #     period=per,
                    #     theta_ic=theta_ic_ca[iper],
                    #     dist=dist_ca[iper],
                    #     pos=pos_ca[iper],
                    #     ls='--',
                    #     label=r'$E_\mathrm{min}=0.2$ TeV, $E_\mathrm{cut}=100$ TeV',
                    #     emin=0.20,
                    #     ecut=100,
                    #     fast=fast_sed
                    # )
                else:
                    fr_mo.plot_sed(
                        iperiod=iper,
                        period=per,
                        theta_ic=theta_ic_mo[iper],
                        dist=dist_mo[iper],
                        pos=pos_mo[iper],
                        ls='-',
                        label=r'$E_\mathrm{min}=0.1$ TeV, $E_\mathrm{cut}=50$ TeV',
                        emin=0.10,
                        ecut=50,
                        fast=fast_sed,
                        best_solution=False,
                        Edot=1e36
                    )
                    # fr_mo.plot_sed(
                    #     iperiod=iper,
                    #     period=per,
                    #     theta_ic=theta_ic_mo[iper],
                    #     dist=dist_mo[iper],
                    #     pos=pos_mo[iper],
                    #     ls='--',
                    #     label=r'$E_\mathrm{min}=0.2$ TeV, $E_\mathrm{cut}=100$ TeV',
                    #     emin=0.20,
                    #     ecut=100,
                    #     fast=fast_sed
                    # )

                data_en, data_fl, data_fl_er = data.get_data(per)
                ax.errorbar(
                    data_en,
                    data_fl,
                    yerr=data_fl_er,
                    linestyle='None',
                    color='k',
                    marker='o'
                )

                data_en_ul, data_fl_ul = data.get_data_ul(per)
                if len(data_en_ul) > 0:
                    data_fl_ul_err = [p - pow(10, math.log10(p)-0.1) for p in data_fl_ul]
                    ax.errorbar(
                        data_en_ul,
                        data_fl_ul,
                        yerr=data_fl_ul_err,
                        uplims=True,
                        color='k',
                        linestyle='none'
                    )

                ax.set_ylim(2e-14, 7e-11)

                if iper == 1:
                    ax.legend(loc='upper left', frameon=False)

            if show:
                plt.show()
            else:
                plt.savefig(
                    'figures/FitSED' + orb + plot_label + '.pdf',
                    format='pdf',
                    bbox_inches='tight'
                )
                plt.savefig(
                    'figures/FitSED' + orb + plot_label + '.png',
                    format='png',
                    bbox_inches='tight'
                )

    def plot_sed_both():
        ##############################
        # SED Both
        plt.figure(figsize=(2 * 8, 2 * 6), tight_layout=True)

        for iper, per in enumerate(periods):
            plt.subplot(2, 2, iper + 1)
            fig = plt.gcf()
            ax = plt.gca()
            ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
            ax.set_xlabel(r'$E$ [keV]')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.tick_params(which='minor', length=minorTickSize)
            ax.tick_params(which='major', length=majorTickSize)

            fermi_spec_en, fermi_spec_fl, fermi_spec_fl_er = data.get_fermi_spec()
            fermi_lim_en, fermi_lim_fl, fermi_lim_fl_er = data.get_fermi_upper_limits()

            # title = 'Nov. 2017' if iper == 0 else 'Dec. 2017'
            # ax.set_title(title)

            # main_ca = round((10**fr_ca.lgEdotMin) / (10**int(fr_ca.lgEdotMin)), 2)
            # pow_ca = int(fr_ca.lgEdotMin)

            # label_ca_sed = (label_ca + '\n' + r'$L_\mathrm{sd}=$' +
            #                 str(main_ca) + r'$\;10^{'+str(pow_ca) + r'}$' +
            #                 r' ergs/s, $\sigma_0$=' + '{:.3f}'.format(10**fr_ca.lgSigmaMin))

            # main_mo = round((10**fr_mo.lgEdotMin) / (10**int(fr_mo.lgEdotMin)), 2)
            # pow_mo = int(fr_mo.lgEdotMin)

            # label_mo_sed = (label_mo + '\n' + r'$L_\mathrm{sd}=$' +
            #                 str(main_mo) + r'$\;10^{'+str(pow_mo) + r'}$' +
            #                 r' ergs/s, $\sigma_0$=' + '{:.3f}'.format(10**fr_mo.lgSigmaMin))

            fr_ca.plot_sed(
                iperiod=iper,
                period=per,
                theta_ic=theta_ic_ca[iper],
                dist=dist_ca[iper],
                pos=pos_ca[iper],
                ls='-',
                # label=label_ca_sed if iper == 0 else None,
                emin=0.10,
                ecut=50,
                fast=fast_sed
                # best_solution=False,
                # Edot=1e36
            )
            fr_mo.plot_sed(
                iperiod=iper,
                period=per,
                theta_ic=theta_ic_mo[iper],
                dist=dist_mo[iper],
                pos=pos_mo[iper],
                ls='--',
                # label=label_mo_sed if iper == 1 else None,
                emin=0.10,
                ecut=50,
                fast=fast_sed,
                # best_solution=False,
                # Edot=1e36
            )

            data_en, data_fl, data_fl_er = data.get_data(per)
            ax.errorbar(
                data_en,
                data_fl,
                yerr=data_fl_er,
                linestyle='None',
                color='k',
                marker='o'
            )

            data_en_ul, data_fl_ul = data.get_data_ul(per)
            if len(data_en_ul) > 0:
                data_fl_ul_err = [p - pow(10, math.log10(p)-0.1) for p in data_fl_ul]
                ax.errorbar(
                    data_en_ul,
                    data_fl_ul,
                    yerr=data_fl_ul_err,
                    uplims=True,
                    color='k',
                    linestyle='none'
                )

            ax.set_ylim(2e-14, 1e-10)

            # if iper == 0:
            ax.legend(loc='upper left', frameon=False)

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitSEDBoth' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitSEDBoth' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

    def plot_mag():
        #################
        # B field - main
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$B_0$ [G]')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=majorTickSize)

        fr_ca.plot_B(line=True, ls='--', label=label_ca, iperiod=0)
        fr_mo.plot_B(line=True, ls=':', label=label_mo, iperiod=0)

        if len(periods) > 2:
            fr_ca.plot_B(line=True, ls='-.', iperiod=3)
            fr_mo.plot_B(line=True, ls='-.', iperiod=3)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        ax.set_ylim(0.04, 0.6)

        ax.legend(frameon=False)

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitBfieldMain' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitBfieldMain' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

    def plot_density():
        #################
        # Density - main
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$U$ [ergs/cm$^3$]')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=majorTickSize)

        fr_ca.plot_density(line=True, ls='--', label=label_ca, iperiod=0)
        fr_mo.plot_density(line=True, ls=':', label=label_mo, iperiod=0)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        ax.legend(frameon=False)

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitDensityMain' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitDensityMain' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

    def plot_dist():
        # Distance - main
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$R_\mathrm{sh, 0}$ [AU]')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=majorTickSize)

        fr_ca.plot_dist(line=True, ls='--', ratio=False, iperiod=0)
        fr_mo.plot_dist(line=True, ls=':', ratio=False, iperiod=0)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitDistanceMain' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitDistanceMain' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

    def plot_opt():
        # Optical Depth - main
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$\tau_{\gamma\gamma,\;0}$')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=7)

        fr_ca.plot_optical_depth(line=True, ls='--', pos=pos_ca[0], iperiod=0)
        fr_mo.plot_optical_depth(line=True, ls=':', pos=pos_mo[0], iperiod=0)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        ax.text(3.2e34, 4.4e-1, r'$E_{\gamma}=0.2$ TeV', horizontalalignment='center')
        ax.text(6e34, 8.3e-2, r'$E_{\gamma}=5.0$ TeV', horizontalalignment='center')

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitOpticalDepthMain' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitOpticalDepthMain' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

    def plot_ebreak():
        # E break - main
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$E_\mathrm{break}$ [TeV]')
        # ax.set_xlabel(r'$\dot{E}$ [erg s$^{-1}$]')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=7)

        ax.axhspan(ymin=0.2, ymax=5, color='k', alpha=0.2)

        fr_ca.plot_ebreak(ls='--')
        fr_mo.plot_ebreak(ls=':')

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        if show:
            plt.show()
        else:
            plt.savefig(
                'figures/FitEbreak' + plot_label + '.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.savefig(
                'figures/FitEbreak' + plot_label + '.png',
                format='png',
                bbox_inches='tight'
            )

    def plot_sig():
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$\sigma$')
        ax.set_xlabel(r'$R_\mathrm{sh}$ [cm]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=7)

        fr_ca.plot_sigma_dist(line=True,
                              ls='--',
                              lw=3,
                              label='This work, orbit by Casares et al. 2012')
        fr_mo.plot_sigma_dist(line=True,
                              ls=':',
                              lw=3,
                              label='This work, orbit by Moritani et al. 2018')

        ms = 10
        # Crab Nebula
        ax.plot([3e17],
                [3e-3],
                color='g',
                linestyle='None',
                marker='^',
                ms=ms,
                label=r'Crab nebula (Kennel \& Coroniti, 1984b)')

        # Crab LC
        ax.plot([1e8],
                [1e5],
                color='g',
                linestyle='None',
                marker='v',
                ms=ms,
                label='Crab - light cylinder (Kong et al. 2012)')

        # PSR B1259-63 LC
        ax.plot([1e8],
                [3e3],
                color='magenta',
                linestyle='None',
                marker='v',
                ms=ms,
                label='PSR B1259-063 - light cylinder (Kong et al. 2012)')

        # PSR B1259-63 Tavani & Arons
        ax.plot([1e12],
                [2e-2],
                color='magenta',
                linestyle='None',
                marker='o',
                ms=ms,
                label=r'PSR B1259-063 - shock (Tavani \& Arons 1997)')

        # LS 5039
        ax.plot([2e10],
                [1e0],
                color='cyan',
                linestyle='None',
                marker='o',
                ms=ms,
                label='LS 5039 - shock (Dubus et al. 2015)')

        # LS 5039
        # ax.plot([3.7e13],
        #         [1e-2],
        #         color='r',
        #         linestyle='None',
        #         marker='o',
        #         ms=ms,
        #         label='PSR J2032+4127 (Dubus et al. 2015)')

        # ax.plot([3e17], [3e-3], color='k', linestyle='None', marker='o')
        # ax.plot([1e8, 1e8], [1e3, 1e4], color='k', linestyle='-', marker='None')

        # ax.text(6e8, 2e3, 'light\ncylinder', horizontalalignment='center')
        # ax.text(3e17, 0.5e-2, 'Crab\nNebula', horizontalalignment='center')

        ax.set_xlim(3e7, 2e18)
        ax.set_ylim(0.12e-2, 0.9e6)

        ax.legend(loc='upper right', frameon=True, fontsize=12)

        if show:
            plt.show()
        else:
            plt.savefig('figures/FitMagnetization' + plot_label + '.pdf', format='pdf', bbox_inches='tight')
            plt.savefig('figures/FitMagnetization' + plot_label + '.png', format='png', bbox_inches='tight')

    def plot_mdot():
        ###########################################
        # Solution - uncertainties m - Casares
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$\sigma_0$')
        # ax.set_xlabel(r'$\dot{E}$ [erg s$^{-1}$]')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(which='minor', length=minorTickSize)
        ax.tick_params(which='major', length=7)

        fr_ca_m_inf.plot_solution(band=band, line=False, ms=40, with_lines=True, no_2s=True,
                                  ls='-', label=r'$\dot{M}_\mathrm{w} = 10^{-9.0}\;M_\odot/\mathrm{yr}$')
        fr_ca.plot_solution(band=band, line=False, ms=40, with_lines=True, no_2s=True,
                            ls='--', label=r'$\dot{M}_\mathrm{w} = 10^{-8.5}\;M_\odot/\mathrm{yr}$')
        fr_ca_m_sup.plot_solution(band=band, line=False, ms=40, with_lines=True, no_2s=True,
                                  ls=':', label=r'$\dot{M}_\mathrm{w} = 10^{-8.0}\;M_\odot/\mathrm{yr}$')

        ax.set_ylim(1e-3, 8.5e-1)

        ax.legend(loc='best', frameon=False)

        if show:
            plt.show()
        else:
            plt.savefig('figures/FitSolutionsCasaresMdot' + plot_label + '.pdf', format='pdf', bbox_inches='tight')
            plt.savefig('figures/FitSolutionsCasaresMdot' + plot_label + '.png', format='png', bbox_inches='tight')

        # Solution - uncertainties m - Moritani
        plt.figure(figsize=(8, 6), tight_layout=True)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_ylabel(r'$\sigma_0$')
        # ax.set_xlabel(r'$\dot{E}$ [erg s$^{-1}$]')
        ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')

        fr_mo_m_inf.plot_solution(band=band, line=False, ms=40, with_lines=True, no_2s=True,
                                  ls='-', label=r'$\dot{M}_\mathrm{w} = 10^{-9.0}\;M_\odot/\mathrm{yr}$')
        fr_mo.plot_solution(band=band, line=False, ms=40, with_lines=True, no_2s=True,
                            ls='--', label=r'$\dot{M}_\mathrm{w} = 10^{-8.5}\;M_\odot/\mathrm{yr}$')
        fr_mo_m_sup.plot_solution(band=band, line=False, ms=40, with_lines=True, no_2s=True,
                                  ls=':', label=r'$\dot{M}_\mathrm{w} = 10^{-8.0}\;M_\odot/\mathrm{yr}$')

        ax.set_ylim(1e-3, 8.5e-1)

        ax.legend(loc='best', frameon=False)

        if show:
            plt.show()
        else:
            plt.savefig('figures/FitSolutionsMoritaniMdot' + plot_label + '.pdf', format='pdf', bbox_inches='tight')
            plt.savefig('figures/FitSolutionsMoritaniMdot' + plot_label + '.png', format='png', bbox_inches='tight')

    if do_solution:
        plot_solution()
    if do_sed:
        plot_sed()
    if do_sed_both:
        plot_sed_both()
    if do_mag:
        plot_mag()
    if do_density:
        plot_density()
    if do_dist:
        plot_dist()
    if do_opt:
        plot_opt()
    if do_ebr:
        plot_ebreak()
    if do_sig:
        plot_sig()
    if do_mdot:
        plot_mdot()
