#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from tgblib.parameters import MJD_MEAN, TSTAR, RSTAR, MJD_0, MONTH_LABEL
from tgblib import util
from tgblib import orbit

if __name__ == '__main__':

    util.set_my_fonts(mode='text')
    show = True
    label = ''

    phases = np.linspace(0.05, 0.95, 10)

    marker = 'x'

    label_ca = 'Casares et al., 2005'
    systems_ca = orbit.getLS5039SystemCasares05()

    label_ar = 'Aragona et al., 2005'
    systems_ar = orbit.getLS5039SystemAragona09()

    label_sa = 'Sarty et al., 2011'
    systems_sa = orbit.getLS5039SystemSarty11()

    for sys in [systems_ca, systems_ar, systems_sa]:
        orbits = orbit.SetOfOrbits(
            phase_step=0.0005,
            color='r',
            systems=sys,
            phases=phases
        )

        pts = orbits.get_pts()
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        orbits.plot_orbit(
            noPoint=True,
            noAxes=True,
            color='r',
            lw=2,
            set_aspect=False,
            only_ref=True
        )
        ax.set_ylabel('Y [AU]')
        ax.set_xlabel('X [AU]')

        x_pos, y_pos = pts['x_pos'], pts['y_pos']
        ph = pts['phase']

        for iper in range(len(phases)):
            ax.plot(
                x_pos[iper],
                y_pos[iper],
                c='k',
                linestyle='None',
                markersize=12,
                marker=marker
            )

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - 0.02, ylim[1] + 0.02)
        ax.set_aspect('equal', adjustable='datalim')

        # ax.text(
        #     0.95,
        #     0.9,
        #     label_ca,
        #     horizontalalignment='right',
        #     transform=ax.transAxes
        # )

        # plt.arrow(2.2, 0, 0, -0.9, width=0.01, head_width=0.18, color='k')
        # plt.text(2.2, 0.2, 'to\n observer', horizontalalignment='center')

        if show:
            plt.show()
        else:
            plt.savefig('figures/OrbitCa' + label + '.png', format='png', bbox_inches='tight')
            plt.savefig('figures/OrbitCa' + label + '.pdf', format='pdf', bbox_inches='tight')
