#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from tgblib.parameters import MJD_MEAN, TSTAR, RSTAR, MJD_0, MONTH_LABEL
from tgblib import util
from tgblib import orbit

if __name__ == '__main__':

    util.set_my_fonts(mode='talk')
    show = False
    label = '_p315'

    periods = [0, 1, 2, 3, 4]

    Tper = 315

    mjd_pts = [MJD_MEAN[p] for p in periods]
    markers = ['o', '*', '^', 's', 'D']
    fillstyle = ['full', 'full', 'full', 'none', 'full']

    extra_label = {
        0: ' - Nu1a + Ve1a',
        1: ' - Nu1b + Ve1b',
        2: ' - Nu2a + Ve2a',
        3: ' - Ve2b',
        4: ' - Nu2b + Ve2c'
    }

    label_ca = 'Casares et al., 2012'
    systems_ca = orbit.generate_systems(
        eccentricity=[0.83],
        phase_per=[0.967],
        inclination=[69.5 * util.degToRad, 59 * util.degToRad, 80 * util.degToRad],
        omega=[129 * util.degToRad],
        period=[Tper],
        mjd_0=[MJD_0],
        temp_star=[TSTAR],
        rad_star=[RSTAR],
        mass_star=[16],
        mass_compact=[1.4],
        f_m=[0.01],
        x1=[0.362]
    )

    label_mo = 'Moritani et al., 2018'
    systems_mo = orbit.generate_systems(
        eccentricity=[0.64],
        phase_per=[0.663],
        inclination=[37 * util.degToRad, 32 * util.degToRad, 42 * util.degToRad],
        omega=[271 * util.degToRad],
        period=[Tper],
        mjd_0=[MJD_0],
        temp_star=[TSTAR],
        rad_star=[RSTAR],
        mass_star=[16],
        mass_compact=[1.4],
        f_m=[0.0024],
        x1=[0.120]
    )

    label_an = 'An, 2020'
    systems_an = orbit.generate_systems(
        eccentricity=[0.45],
        phase_per=[0.3],
        inclination=[47 * util.degToRad],
        omega=[279 * util.degToRad],
        period=[Tper],
        mjd_0=[MJD_0],
        temp_star=[TSTAR],
        rad_star=[RSTAR],
        mass_star=[16],
        mass_compact=[1.4],
        x1=[0.190663]
    )

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
    orbits_an = orbit.SetOfOrbits(
        phase_step=0.0005,
        color='g',
        systems=systems_an,
        mjd_pts=mjd_pts
    )

    pts_ca = orbits_ca.get_pts()
    pts_mo = orbits_mo.get_pts()
    pts_an = orbits_an.get_pts()
    x_pos_an, y_pos_an = pts_an['x_pos'], pts_an['y_pos']

    print(pts_an['phase'])

    ##########
    # Orbit Ca
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_orbit(
        noPoint=True,
        noAxes=True,
        color='r',
        lw=2,
        set_aspect=False,
        only_ref=True
    )
    ax.set_ylabel('Y [AU]')
    ax.set_xlabel('X [AU]')

    x_pos_ca, y_pos_ca = pts_ca['x_pos'], pts_ca['y_pos']
    ph_ca = pts_ca['phase']

    for iper in range(len(periods)):
        ax.plot(
            x_pos_ca[iper],
            y_pos_ca[iper],
            c='k',
            linestyle='None',
            markersize=12,
            fillstyle=fillstyle[iper],
            marker=markers[iper],
            label=MONTH_LABEL[iper] + extra_label[iper]
        )

    disk_ca = plt.Circle(
        (0, 0),
        1.12,
        color='black',
        fill=False,
        lw=2,
        ls=':'
    )
    ax.add_artist(disk_ca)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.2, ylim[1] + 0.2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, loc='lower left', handletextpad=0.7, handlelength=0)

    ax.text(
        0.95,
        0.9,
        label_ca,
        horizontalalignment='right',
        transform=ax.transAxes
    )

    plt.arrow(2.2, 0, 0, -0.9, width=0.01, head_width=0.18, color='k')
    plt.text(2.2, 0.2, 'to\n observer', horizontalalignment='center')

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitCa' + label + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitCa' + label + '.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Orbit Mo
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_mo.plot_orbit(
        noPoint=True,
        noAxes=True,
        color='b',
        lw=2,
        set_aspect=False,
        only_ref=True
    )
    ax.set_ylabel('Y [AU]')
    ax.set_xlabel('X [AU]')

    x_pos_mo, y_pos_mo = pts_mo['x_pos'], pts_mo['y_pos']
    ph_mo = pts_mo['phase']

    for iper in range(len(periods)):
        ax.plot(
            x_pos_mo[iper],
            y_pos_mo[iper],
            c='k',
            linestyle='None',
            markersize=12,
            marker=markers[iper],
            fillstyle=fillstyle[iper],
            label=MONTH_LABEL[iper] + extra_label[iper]
        )

    disk_mo = plt.Circle(
        (0, 0),
        1.12,
        color='black',
        fill=False,
        lw=2,
        ls=':'
    )
    ax.add_artist(disk_mo)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + 0.9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, loc='upper left', handletextpad=0.7, handlelength=0)

    ax.text(
        0.95,
        0.9,
        label_mo,
        horizontalalignment='right',
        transform=ax.transAxes
    )

    plt.arrow(3, -2.4, 0, -0.7, width=0.01, head_width=0.18, color='k')
    plt.text(3, -2.2, 'to\n observer', horizontalalignment='center')

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitMo' + label + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitMo' + label + '.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Orbit both
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_orbit(
        noPoint=True,
        noAxes=True,
        lw=2,
        set_aspect=False,
        only_ref=True
    )
    orbits_mo.plot_orbit(
        noPoint=True,
        noAxes=True,
        lw=2,
        set_aspect=False,
        only_ref=True
    )
    orbits_an.plot_orbit(
        noPoint=True,
        noAxes=True,
        lw=2,
        set_aspect=False,
        only_ref=True
    )
    ax.set_ylabel('Y [AU]')
    ax.set_xlabel('X [AU]')

    for iper in range(len(periods)):
        ax.plot(
            x_pos_ca[iper],
            y_pos_ca[iper],
            c='k',
            linestyle='None',
            markersize=10,
            fillstyle=fillstyle[iper],
            marker=markers[iper],
            label=MONTH_LABEL[iper] + extra_label[iper]
        )

        ax.plot(
            x_pos_mo[iper],
            y_pos_mo[iper],
            c='k',
            linestyle='None',
            markersize=10,
            fillstyle=fillstyle[iper],
            marker=markers[iper]
        )

        ax.plot(
            x_pos_an[iper],
            y_pos_an[iper],
            c='k',
            linestyle='None',
            markersize=10,
            fillstyle=fillstyle[iper],
            marker=markers[iper]
        )

    disk = plt.Circle(
        (0, 0),
        1.12,
        color='black',
        fill=False,
        lw=2,
        ls=':'
    )

    ax.add_artist(disk)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.1, ylim[1] + 0.3)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1, xlim[1] + 1.0)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, loc='upper right', fontsize=14, handletextpad=0.7, handlelength=0)

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitBoth' + label + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitBoth' + label + '.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Distance

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_distance(
        noPoint=True,
        all_lines=False,
        noAxes=True,
        color='r',
        ls='-',
        only_ref=False,
        alpha=0.1,
        label=label_ca
    )
    orbits_mo.plot_distance(
        noPoint=True,
        all_lines=False,
        noAxes=True,
        color='b',
        ls='--',
        only_ref=False,
        alpha=0.1,
        label=label_mo
    )

    disk_x = np.linspace(0, 1, 100)
    disk_y = [1.12 for i in range(len(disk_x))]
    ax.plot(disk_x, disk_y, linestyle=':', color='k')

    dist_ca, ph_ca = pts_ca['distance'], pts_ca['phase']
    dist_mo, ph_mo = pts_mo['distance'], pts_mo['phase']

    for iper in range(len(periods)):
        ax.plot(
            ph_ca[iper],
            dist_ca[iper],
            c='k',
            linestyle='None',
            markersize=12,
            fillstyle=fillstyle[iper],
            marker=markers[iper],
            label=MONTH_LABEL[iper] + extra_label[iper]
        )

        ax.plot(
            ph_mo[iper],
            dist_mo[iper],
            c='k',
            linestyle='None',
            markersize=12,
            marker=markers[iper],
            fillstyle=fillstyle[iper],
            label=MONTH_LABEL[iper] + extra_label[iper]
        )

    # handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    # leg0 = plt.legend(handles[0:2], labels[0:2], loc='upper left', frameon=False)
    # leg1 = plt.legend(handles[2:4], labels[2:4], loc='upper right', frameon=False)
    # ax.add_artist(leg0)
    # ax.add_artist(leg1)

    ax.set_ylabel('D [AU]')
    ax.set_xlabel(r'Orbital phase, $\phi$')

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 8.5)

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitDistance' + label + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitDistance' + label + '.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Theta IC

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_theta_scat(
        noPoint=True,
        all_lines=False,
        noAxes=True,
        color='r',
        ls='-',
        only_ref=False,
        alpha=0.1,
        label=label_ca
    )
    orbits_mo.plot_theta_scat(
        noPoint=True,
        all_lines=False,
        noAxes=True,
        color='b',
        ls='--',
        only_ref=False,
        alpha=0.1,
        label=label_mo
    )

    theta_ca = pts_ca['theta_ic']
    theta_mo = pts_mo['theta_ic']
    for iper in range(len(periods)):
        ax.plot(
            ph_ca[iper],
            theta_ca[iper],
            c='k',
            linestyle='None',
            markersize=12,
            marker=markers[iper],
            fillstyle=fillstyle[iper],
            label=MONTH_LABEL[iper] + extra_label[iper]
        )
        ax.plot(
            ph_mo[iper],
            theta_mo[iper],
            c='k',
            linestyle='None',
            markersize=12,
            marker=markers[iper],
            fillstyle=fillstyle[iper],
            label=MONTH_LABEL[iper] + extra_label[iper]
        )

    handles, labels = ax.get_legend_handles_labels()
    leg0 = plt.legend(handles[0:2], labels[0:2], loc='lower left', frameon=False)
    leg1 = plt.legend(handles[2:4], labels[2:4], loc='upper right', frameon=False)
    ax.add_artist(leg0)
    ax.add_artist(leg1)

    ax.set_ylabel(r'$\theta_\mathrm{ICS} [^{\circ}]$')
    ax.set_xlabel(r'Orbital phase, $\phi$')

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitTheta' + label + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitTheta' + label + '.pdf', format='pdf', bbox_inches='tight')
