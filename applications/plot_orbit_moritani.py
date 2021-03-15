#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from tgblib.parameters import MJD_MEAN, TSTAR, RSTAR, MJD_0, MONTH_LABEL
from tgblib import util
from tgblib import orbit

if __name__ == '__main__':

    util.set_my_fonts(mode='text')
    show = False
    label = '_moritaniBoth'

    periods = [0, 1, 2, 3, 4]

    Tper = 315

    mjd_pts = [MJD_MEAN[p] for p in periods]
    markers = ['o', '*', '^', 's', 'D']
    fillstyle = ['full', 'full', 'full', 'none', 'full']

    extra_label = {
        0: '',
        1: '',
        2: '',
        3: ' (only VTS)',
        4: ''
    }

    label_1 = '313 days'
    systems_1 = orbit.generate_systems(
        eccentricity=[0.64],
        phase_per=[0.663],
        inclination=[37 * util.degToRad],
        omega=[271 * util.degToRad],
        period=[313],
        mjd_0=[MJD_0],
        temp_star=[TSTAR],
        rad_star=[RSTAR],
        mass_star=[16],
        mass_compact=[1.4],
        f_m=[0.0024],
        x1=[0.120]
    )

    orbits_1 = orbit.SetOfOrbits(
        phase_step=0.0005,
        color='r',
        systems=systems_1,
        mjd_pts=mjd_pts
    )

    label_2 = '308 days'
    systems_2 = orbit.generate_systems(
        eccentricity=[0.62],
        phase_per=[0.709],
        inclination=[37 * util.degToRad],
        omega=[249 * util.degToRad],
        period=[308],
        mjd_0=[MJD_0],
        temp_star=[TSTAR],
        rad_star=[RSTAR],
        mass_star=[16],
        mass_compact=[1.4],
        f_m=[0.0035],
        x1=[0.136]
    )

    orbits_1 = orbit.SetOfOrbits(
        phase_step=0.0005,
        color='r',
        systems=systems_1,
        mjd_pts=mjd_pts
    )
    orbits_2 = orbit.SetOfOrbits(
        phase_step=0.0005,
        color='b',
        systems=systems_2,
        mjd_pts=mjd_pts
    )

    pts_1 = orbits_1.get_pts()
    pts_2 = orbits_2.get_pts()

    x_pos_1, y_pos_1 = pts_1['x_pos'], pts_1['y_pos']
    ph_1 = pts_1['phase']
    x_pos_2, y_pos_2 = pts_2['x_pos'], pts_2['y_pos']
    ph_2 = pts_2['phase']

    ##########
    # Orbit both
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_1.plot_orbit(
        noPoint=True,
        noAxes=True,
        lw=2,
        set_aspect=False,
        only_ref=True
    )
    orbits_2.plot_orbit(
        noPoint=True,
        noAxes=True,
        lw=2,
        set_aspect=False,
        only_ref=True
    )
    ax.set_ylabel('Y [AU]')
    ax.set_xlabel('X [AU]')

    # disk = plt.Circle(
    #     (0, 0),
    #     1.12,
    #     color='black',
    #     fill=False,
    #     lw=2,
    #     ls=':'
    # )

    # ax.add_artist(disk)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.2, ylim[1] + 0.2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, loc='lower left', handletextpad=0.7, handlelength=0)

    if show:
        plt.show()
    else:
        plt.savefig('figures/Orbit' + label + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/Orbit' + label + '.pdf', format='pdf', bbox_inches='tight')
