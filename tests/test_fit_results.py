#!/usr/bin/python3

import matplotlib.pyplot as plt

import tgblib.fit_results as fr


def test_init():
    fr_mo = fr.FitResult(
        n_periods=2,
        label='test',
        color='r',
        SigmaMax=1e5,
        EdotMin=1e34
    )


def test_plot_solution():
    fr_mo = fr.FitResult(
        n_periods=4,
        label='test',
        color='r',
        SigmaMax=1e5,
        EdotMin=1e34
    )

    # Solution - main
    plt.figure(figsize=(8, 6), tight_layout=True)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_ylabel(r'$\sigma_0$')
    # ax.set_xlabel(r'$\dot{E}$ [erg s$^{-1}$]')
    ax.set_xlabel(r'$L_\mathrm{sd}$ [erg s$^{-1}$]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.tick_params(which='minor', length=minorTickSize)
    # ax.tick_params(which='major', length=majorTickSize)

    fr_mo.plot_solution(
        band=True,
        line=True,
        ms=40,
        with_lines=True,
        no_2s=True,
        ls='-',
        line_ls=':',
        label='test'
    )

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ylim = [1e-3, 1e-1]
    ax.set_ylim(ylim[0], ylim[1])

    plt.show()


if __name__ == '__main__':

    # test_init()
    test_plot_solution()
