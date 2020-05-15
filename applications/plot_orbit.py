#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import itertools
from itertools import combinations
from numpy import arccos
from numpy.linalg import norm

import astropy.constants as const
import astropy.units as u

from tgblib import util
from tgblib import orbit

if __name__ == '__main__':

    util.set_my_fonts(mode='talk')
    show = False

    label_nov = 'Nov. 2017'
    label_dec = 'Dec. 2017'

    label_ca = 'Casares et al., 2012'
    systems_ca = orbit.generate_systems(eccentricity=[0.83],
                                        phase_per=[0.967],
                                        inclination=[69.5 * util.degToRad,
                                                     59 * util.degToRad,
                                                     80 * util.degToRad],
                                        omega=[129 * util.degToRad],
                                        period=[315],
                                        mjd_0=[54857.5],
                                        temp_star=[30e3],
                                        rad_star=[7.8],
                                        mass_star=[16],
                                        mass_compact=[1.4],
                                        f_m=[0.01],
                                        x1=[0.362])

    label_mo = 'Moritani et al., 2018'
    systems_mo = orbit.generate_systems(eccentricity=[0.64],
                                        phase_per=[0.663],
                                        inclination=[37 * util.degToRad,
                                                     32 * util.degToRad,
                                                     42 * util.degToRad],
                                        omega=[271 * util.degToRad],
                                        period=[315],
                                        mjd_0=[54857.5],
                                        temp_star=[30e3],
                                        rad_star=[7.8],
                                        mass_star=[16],
                                        mass_compact=[1.4],
                                        f_m=[0.0024],
                                        x1=[0.120])
    mjd_pts = [58078, 58101]
    orbits_ca = orbit.SetOfOrbits(phase_step=0.0005,
                                  color='r',
                                  systems=systems_ca,
                                  mjd_pts=mjd_pts)
    orbits_mo = orbit.SetOfOrbits(phase_step=0.0005,
                                  color='b',
                                  systems=systems_mo,
                                  mjd_pts=mjd_pts)

    pts_ca = orbits_ca.get_pts()
    pts_mo = orbits_mo.get_pts()

    print(pts_ca)
    print('------')
    print(pts_mo)

    ##########
    # Orbit Ca
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_orbit(noPoint=True,
                         noAxes=True,
                         color='r',
                         lw=2,
                         set_aspect=False,
                         only_ref=True)
    ax.set_ylabel('Y [AU]')
    ax.set_xlabel('X [AU]')

    x_pos_ca, y_pos_ca = pts_ca['x_pos'], pts_ca['y_pos']
    ph_ca = pts_ca['phase']

    ax.plot(x_pos_ca[0],
            y_pos_ca[0],
            c='k',
            linestyle='None',
            markersize=12,
            marker='o',
            label=label_nov)
    ax.plot(x_pos_ca[1],
            y_pos_ca[1],
            c='k',
            linestyle='None',
            markersize=15,
            marker='*',
            label=label_dec)

    disk_ca = plt.Circle((0, 0), 1.12,
                         color='black', fill=False, lw=2, ls=':')
    ax.add_artist(disk_ca)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.2, ylim[1] + 0.2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, loc='lower left', handletextpad=0.7, handlelength=0)

    ax.text(0.95, 0.9, label_ca,
            horizontalalignment='right', transform=ax.transAxes)

    plt.arrow(2.2, 0, 0, -0.9, width=0.01, head_width=0.18, color='k')
    plt.text(2.2, 0.2, 'to\n observer', horizontalalignment='center')

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitCa.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitCa.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Orbit Mo
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_mo.plot_orbit(noPoint=True, noAxes=True, color='b', lw=2,
                         set_aspect=False, only_ref=True)
    ax.set_ylabel('Y [AU]')
    ax.set_xlabel('X [AU]')

    x_pos_mo, y_pos_mo = pts_mo['x_pos'], pts_mo['y_pos']
    ph_mo = pts_mo['phase']

    ax.plot(x_pos_mo[0],
            y_pos_mo[0],
            c='k',
            linestyle='None',
            markersize=12,
            marker='o',
            label=label_nov)
    ax.plot(x_pos_mo[1],
            y_pos_mo[1],
            c='k',
            linestyle='None',
            markersize=15,
            marker='*',
            label=label_dec)

    disk_mo = plt.Circle((0, 0), 1.12,
                         color='black', fill=False, lw=2, ls=':')
    ax.add_artist(disk_mo)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + 0.9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, loc='upper left', handletextpad=0.7, handlelength=0)

    ax.text(0.95, 0.9, label_mo,
            horizontalalignment='right', transform=ax.transAxes)

    plt.arrow(3, -2.4, 0, -0.7, width=0.01, head_width=0.18, color='k')
    plt.text(3, -2.2, 'to\n observer', horizontalalignment='center')

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitMo.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitMo.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Orbit both
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_orbit(noPoint=True,
                         noAxes=True,
                         lw=2,
                         set_aspect=False,
                         only_ref=True)
    orbits_mo.plot_orbit(noPoint=True,
                         noAxes=True,
                         lw=2,
                         set_aspect=False,
                         only_ref=True)
    ax.set_ylabel('Y [AU]')
    ax.set_xlabel('X [AU]')

    ax.plot(x_pos_ca[0],
            y_pos_ca[0],
            c='k',
            linestyle='None',
            markersize=12,
            marker='o',
            label=label_nov)
    ax.plot(x_pos_ca[1],
            y_pos_ca[1],
            c='k',
            linestyle='None',
            markersize=15,
            marker='*',
            label=label_dec)

    ax.plot(x_pos_mo[0],
            y_pos_mo[0],
            c='k',
            linestyle='None',
            markersize=12,
            marker='o')
    ax.plot(x_pos_mo[1],
            y_pos_mo[1],
            c='k',
            linestyle='None',
            markersize=15,
            marker='*')

    disk = plt.Circle((0, 0), 1.12,
                      color='black', fill=False, lw=2, ls=':')

    ax.add_artist(disk)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.2, ylim[1] + 0.2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False, loc='lower left', handletextpad=0.7, handlelength=0)

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitBoth.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitBoth.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Distance

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_distance(noPoint=True,
                            all_lines=False,
                            noAxes=True,
                            color='r',
                            ls='-',
                            only_ref=False,
                            alpha=0.1,
                            label=label_ca)
    orbits_mo.plot_distance(noPoint=True,
                            all_lines=False,
                            noAxes=True,
                            color='b',
                            ls='--',
                            only_ref=False,
                            alpha=0.1,
                            label=label_mo)

    disk_x = np.linspace(0, 1, 100)
    disk_y = [1.12 for i in range(len(disk_x))]
    ax.plot(disk_x, disk_y, linestyle=':', color='k')

    dist_ca, ph_ca = pts_ca['distance'], pts_ca['phase']
    ax.plot(ph_ca[0], dist_ca[0], c='k', linestyle='None',
            markersize=12, marker='o', label=label_nov)
    ax.plot(ph_ca[1], dist_ca[1], c='k', linestyle='None',
            markersize=15, marker='*', label=label_dec)

    dist_mo, ph_mo = pts_mo['distance'], pts_mo['phase']
    ax.plot(ph_mo[0], dist_mo[0], c='k', linestyle='None',
            markersize=12, marker='o', label=label_nov)
    ax.plot(ph_mo[1], dist_mo[1], c='k', linestyle='None',
            markersize=15, marker='*', label=label_dec)

    handles, labels = ax.get_legend_handles_labels()
    print(labels)
    leg0 = plt.legend(handles[0:2], labels[0:2], loc='upper left', frameon=False)
    leg1 = plt.legend(handles[2:4], labels[2:4], loc='upper right', frameon=False)
    ax.add_artist(leg0)
    ax.add_artist(leg1)

    ax.set_ylabel('D [AU]')
    ax.set_xlabel(r'Orbital phase, $\phi$')

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 8.5)

    if show:
        plt.show()
    else:
        plt.savefig('figures/OrbitDistance.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitDistance.pdf', format='pdf', bbox_inches='tight')

    ##########
    # Theta IC

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    orbits_ca.plot_theta_scat(noPoint=True, all_lines=False, noAxes=True, color='r', ls='-',
                              only_ref=False, alpha=0.1, label=label_ca)
    orbits_mo.plot_theta_scat(noPoint=True, all_lines=False, noAxes=True, color='b', ls='--',
                              only_ref=False, alpha=0.1, label=label_mo)

    theta_ca = pts_ca['theta_ic']
    ax.plot(ph_ca[0], theta_ca[0], c='k', linestyle='None',
            markersize=12, marker='o', label=label_nov)
    ax.plot(ph_ca[1], theta_ca[1], c='k', linestyle='None',
            markersize=15, marker='*', label=label_dec)

    theta_mo = pts_mo['theta_ic']
    ax.plot(ph_mo[0], theta_mo[0], c='k', linestyle='None',
            markersize=12, marker='o', label=label_nov)
    ax.plot(ph_mo[1], theta_mo[1], c='k', linestyle='None',
            markersize=15, marker='*', label=label_dec)

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
        plt.savefig('figures/OrbitTheta.png', format='png', bbox_inches='tight')
        plt.savefig('figures/OrbitTheta.pdf', format='pdf', bbox_inches='tight')
