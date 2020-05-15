#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import copy
from numpy import arccos
from numpy.linalg import norm
from itertools import combinations

import astropy.constants as const
import astropy.units as u

from tgblib import util


class SystemParameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        required_par = ['period', 'eccentricity', 'inclination',
                        'omega', 'mass_star', 'mjd_0', 'mass_compact',
                        'temp_star', 'rad_star', 'phase_per', 'x1', 'f_m']
        for p in required_par:
            if p not in kwargs.keys():
                print('Error: SystemParameters does not contain', p)
        if 'is_ref' not in kwargs.keys():
            self.__dict__['is_ref'] = False
        if 'f_m' not in kwargs.keys():
            self.__dict__['f_m'] = None


class SetOfOrbits(object):
    def __init__(self, phase_step=0.001,
                 color='b', systems=None, mjd_pts=None, mjd_ph_0=None):
        self.sys = systems if isinstance(systems, list) else [systems]
        self.mjd_ph_0 = mjd_ph_0
        self.color = color
        self.n_sys = len(self.sys)
        self.orbits = list()
        for s in self.sys:
            o = Orbit(s, phase_step=phase_step)
            if mjd_pts is not None:
                o.set_points(mjd_pts=mjd_pts)
            self.orbits.append(o)
            if s.is_ref:
                self.orbit_ref = o

        self.compute_mjd_phase()

        self.phase_band, self.distance_band_hi, self.distance_band_lo = list(), list(), list()
        self.theta_band_hi, self.theta_band_lo = list(), list()
        self.density_band_hi, self.density_band_lo = list(), list()

        # Band
        for i in range(len(self.orbits[0].phase)):
            max_dist = self.orbits[0].radius[i]
            min_dist = self.orbits[0].radius[i]
            max_theta = self.orbits[0].theta_scat[i]
            min_theta = self.orbits[0].theta_scat[i]
            max_density = self.orbits[0].density[i]
            min_density = self.orbits[0].density[i]

            for o in self.orbits:
                max_dist = max(max_dist, o.radius[i])
                min_dist = min(min_dist, o.radius[i])
                max_theta = max(max_theta, o.theta_scat[i])
                min_theta = min(min_theta, o.theta_scat[i])
                max_density = max(max_density, o.density[i])
                min_density = min(min_density, o.density[i])

            self.phase_band.append(self.orbits[0].phase[i])
            self.distance_band_hi.append(max_dist)
            self.distance_band_lo.append(min_dist)
            self.theta_band_hi.append(max_theta)
            self.theta_band_lo.append(min_theta)
            self.density_band_hi.append(max_density)
            self.density_band_lo.append(min_density)

    def compute_mjd_phase(self):
        if self.mjd_ph_0 is None:
            print('No mjd_ph_0 input')
            self.mjd_phase = None
            return None

        ph = self.orbit_ref.phase
        mjd_0 = self.orbit_ref.mjd_0
        per = self.orbit_ref.period
        mjd_step = per / (len(ph) - 1)

        ph_0 = (self.mjd_ph_0 - mjd_0) / per
        ph_0 -= int(ph_0)

        self.mjd_phase = [0] * len(ph)
        idx_0 = np.argmin(np.array([(p-ph_0)**2 for p in ph]))
        for i in range(len(ph)):
            idx = idx_0 + i
            if idx > len(ph)-1:
                idx -= len(ph)
            self.mjd_phase[idx] = self.mjd_ph_0 + i * mjd_step

    def plot_orbit(self, noPoint=False, only_ref=False, noAxes=False,
                   color=None, lw=None, set_aspect=True):
        ax = plt.gca()
        cc = color if color is not None else self.color
        if not noAxes:
            ax.tick_params(labelsize=22)
            ax.set_ylabel('Y [AU]', fontsize=22)
            ax.set_xlabel('X [AU]', fontsize=22)

        plt.plot([0], [0], marker='o', c='k', linestyle='None', markersize=9)
        if not only_ref:
            for i in range(self.n_sys):
                plt.plot(self.orbits[i].posX, self.orbits[i].posY, linestyle='-',
                         marker='None', c=cc)
                if self.orbits[i].n_pts > 0 and not noPoint:
                    plt.plot(self.orbits[i].posX_pts, self.orbits[i].posY_pts,
                             linestyle='None', marker='o', c='g', markersize=7)
        else:
            plt.plot(self.orbit_ref.posX, self.orbit_ref.posY, linestyle='-',
                     linewidth=lw, marker='None', c=cc)

        if self.orbit_ref.n_pts > 0 and not noPoint:
            plt.plot(self.orbit_ref.posX_pts, self.orbit_ref.posY_pts,
                     linestyle='None', marker='o', c='g', markersize=10)

        if set_aspect:
            ax.set_aspect('equal', adjustable='datalim')

    def plot_distance(self, noPoint=False, noAxes=False, only_ref=False, band=True,
                      all_lines=True, in_mjd=False,
                      color=None, ls=None, label='None', alpha=None):
        ax = plt.gca()
        cc = color if color is not None else self.color
        ls = ls if ls is not None else '-'
        aa = alpha if alpha is not None else 0.4
        if not noAxes:
            ax.tick_params(labelsize=22)
            ax.set_ylabel('distance [AU]', fontsize=22)
            ax.set_xlabel(r'$\phi$', fontsize=22)

        if not only_ref:
            if band:
                x_band0 = self.phase_band if not in_mjd else self.mjd_phase
                x_band, hi_band, lo_band = zip(*sorted(zip(x_band0,
                                                           self.distance_band_hi,
                                                           self.distance_band_lo)))
                ax.fill_between(x_band, lo_band, hi_band,
                                color=cc, alpha=aa)
            if all_lines:
                for i in range(self.n_sys):
                    ll = label if i == 0 else None
                    x_plot0 = self.orbits[i].phase if not in_mjd else self.mjd_phase
                    x_plot, rad_plot = zip(*sorted(zip(x_plot0,
                                                       self.orbits[i].radius)))
                    plt.plot(x_plot, rad_plot, linestyle=ls, marker='None',
                             c=cc, markersize=2, label=ll)
                    if self.orbits[i].n_pts > 0 and not noPoint:
                        plt.plot(self.orbits[i].phase_pts, self.orbits[i].radius_pts,
                                 linestyle='None', marker='o', c='g', markersize=10)

            x_plot0 = self.orbit_ref.phase if not in_mjd else self.mjd_phase
            x_plot, rad_plot = zip(*sorted(zip(x_plot0, self.orbit_ref.radius)))

            plt.plot(x_plot, rad_plot, linestyle=ls, marker='None',
                     linewidth=2, c=cc, markersize=2, label=label)

        else:
            x_plot0 = self.orbit_ref.phase if not in_mjd else self.mjd_phase
            x_plot, rad_plot = zip(*sorted(zip(x_plot0, self.orbit_ref.radius)))

            plt.plot(x_plot, rad_plot, linestyle=ls, marker='None',
                     linewidth=2, c=cc, markersize=2, label=label)

        if self.orbit_ref.n_pts > 0 and not noPoint:
            plt.plot(self.orbit_ref.phase_pts, self.orbit_ref.radius_pts,
                     linestyle='None', marker='o', c='g', markersize=10)

    def plot_theta_scat(self, noPoint=False, noAxes=False, label='None',
                        only_ref=False, band=True, all_lines=True, color=None, ls=None, alpha=None):
        ax = plt.gca()
        cc = color if color is not None else self.color
        ls = ls if ls is not None else '-'
        aa = alpha if alpha is not None else 0.4
        if not noAxes:
            ax.tick_params(labelsize=22)
            ax.set_ylabel(r'$\theta_\mathrm{IC}$ [deg]', fontsize=22)
            ax.set_xlabel(r'$\phi$', fontsize=22)

        if not only_ref:
            if band:
                ph_band, hi_band, lo_band = zip(*sorted(zip(self.phase_band,
                                                            self.theta_band_hi,
                                                            self.theta_band_lo)))
                ax.fill_between(ph_band, [t / util.degToRad for t in lo_band],
                                [t / util.degToRad for t in hi_band],
                                color=cc, alpha=aa)
            if all_lines:
                for i in range(self.n_sys):
                    ll = label if i == 0 else None
                    ph_plot, th_plot = zip(*sorted(zip(self.orbits[i].phase,
                                                       self.orbits[i].theta_scat)))
                    plt.plot(ph_plot, [t / util.degToRad for t in th_plot],
                             linestyle=ls, marker='None',
                             c=cc, markersize=2, label=ll)
                    if self.orbits[i].n_pts > 0 and not noPoint:
                        plt.plot(self.orbits[i].phase_pts,
                                 [t / util.degToRad for t in self.orbits[i].theta_scat_pts],
                                 linestyle='None', marker='o', c='g', markersize=10)

            ph_plot, th_plot = zip(*sorted(zip(self.orbit_ref.phase,
                                           self.orbit_ref.theta_scat)))
            plt.plot(ph_plot, [t / util.degToRad for t in th_plot],
                     linestyle=ls, marker='None', c=cc,
                     linewidth=2, markersize=2, label=label)

        else:
            ph_plot, th_plot = zip(*sorted(zip(self.orbit_ref.phase,
                                               self.orbit_ref.theta_scat)))
            plt.plot(ph_plot, [t / util.degToRad for t in th_plot],
                     linestyle=ls, marker='None', c=cc,
                     linewidth=2, markersize=2, label=label)

        if self.orbit_ref.n_pts > 0 and not noPoint:
            plt.plot(self.orbit_ref.phase_pts,
                     [t / util.degToRad for t in self.orbit_ref.theta_scat_pts],
                     linestyle='None', marker='o', c='g', markersize=10)

    def plot_density(self, noPoint=False, only_ref=False, band=True, all_lines=True):
        ax = plt.gca()
        ax.tick_params(labelsize=22)
        ax.set_ylabel('U [erg/cm³]', fontsize=22)
        ax.set_xlabel(r'$\phi$', fontsize=22)
        ax.set_yscale('log')

        if not only_ref:
            if band:
                ph_band, hi_band, lo_band = zip(*sorted(zip(self.phase_band,
                                                            self.density_band_hi,
                                                            self.density_band_lo)))
                ax.fill_between(ph_band, lo_band, hi_band,
                                color=self.color, alpha=0.4)

            if all_lines:
                for i in range(self.n_sys):
                    ph_plot, d_plot = zip(*sorted(zip(self.orbits[i].phase,
                                                      self.orbits[i].density)))
                    plt.plot(ph_plot, d_plot,
                             linestyle='-', marker='None', c=self.color, markersize=2)
                    if self.orbits[i].n_pts > 0 and not noPoint:
                        plt.plot(self.orbits[i].phase_pts, self.orbits[i].density_pts,
                                 linestyle='None', marker='o', c='g', markersize=10)

        ph_plot, d_plot = zip(*sorted(zip(self.orbit_ref.phase, self.orbit_ref.density)))
        plt.plot(ph_plot, d_plot,
                 linestyle='-', marker='None', c=self.color,
                 linewidth=3, markersize=2)
        if self.orbit_ref.n_pts > 0 and not noPoint:
            plt.plot(self.orbit_ref.phase_pts, self.orbit_ref.density_pts,
                     linestyle='None', marker='o', c='g', markersize=10)

    def print_pts(self):
        for i in range(self.orbit_ref.n_pts):
            print('======')
            print('Pt ', i)
            print('Distance = {:.3f}'.format(self.orbit_ref.radius_pts[i]))
            print('ThetaIC = {:.3f}'.format(self.orbit_ref.theta_scat_pts[i] / util.degToRad))
            print('Density = {:.3f}'.format(self.orbit_ref.density_pts[i]))

    def get_pts(self):
        pts = dict()
        pts['phase'] = self.orbit_ref.phase_pts
        pts['distance'] = self.orbit_ref.radius_pts
        pts['theta_ic'] = [t / util.degToRad for t in self.orbit_ref.theta_scat_pts]
        pts['density'] = self.orbit_ref.density_pts
        pts['x_pos'] = self.orbit_ref.posX_pts
        pts['y_pos'] = self.orbit_ref.posY_pts
        pts['pos_3D'] = self.orbit_ref.pos3D_pts
        return pts


class Orbit(object):
    def __init__(self, system, phase_step=0.001):
        if not isinstance(system, SystemParameters):
            print('Orbit: system is not an instance of SystemParameters')
        self.phase_step = phase_step
        self.mjd_0 = system.mjd_0
        self.eccentricity = system.eccentricity
        self.omega = system.omega
        self.inclination = system.inclination
        self.phase_per = system.phase_per
        self.period = system.period
        self.mass_compact = system.mass_compact
        self.mass_star = self.compute_mass_star(system.f_m, self.mass_compact, self.inclination)\
            if system.f_m is not None else system.mass_star
        self.mass_ratio = self.mass_star / self.mass_compact
        self.a = self.mass_ratio * system.x1 / math.sin(self.inclination)
        self.apastron = self.a * (1 + self.eccentricity)
        self.periastron = self.a * (1 - self.eccentricity)
        self.area = (self.a**2) * math.sqrt(1 - self.eccentricity**2) * math.pi
        self.rstar = system.rad_star
        self.tstar = system.temp_star

        # print('a=', self.a, ' i=', self.inclination)

        self.phase, self.distance = np.array([]), np.array([])
        self.posX, self.posY = np.array([]), np.array([])
        self.posX3D, self.posY3D, self.posZ3D = np.array([]), np.array([]), np.array([])
        self.theta, self.radius = np.array([]), np.array([])
        self.theta_scat, self.density = np.array([]), np.array([])
        self.n_pts = 0

        self.run()

    def compute_mass_star(self, f, m_c, i):
        x = (m_c**3) * (math.sin(i)**3) / f
        return math.sqrt(x) - m_c

    def run(self):
        ph = self.phase_per + 0.5
        ph = ph - int(ph)
        rad = self.apastron
        th = 0
        x, y = rad * math.cos(th + self.omega), rad * math.sin(th + self.omega)
        x3, y3, z3 = x, y * math.cos(self.inclination), y * math.sin(self.inclination)
        rad_vec = np.array([x3, y3, z3])
        obs_vec = np.array([0, 0, -1])
        th_sc = arccos(np.dot(rad_vec, obs_vec) / norm(rad_vec))

        d_norm = (const.sigma_sb / const.c)
        d_norm *= (self.tstar * u.K)**4
        d_norm *= (self.rstar * u.Rsun)**2
        d = (d_norm / (rad * u.au)**2).to(u.erg / u.cm**3).value

        for i in range(int(1. / self.phase_step) + 1):
            self.phase = np.append(self.phase, ph)
            self.radius = np.append(self.radius, rad)
            self.theta = np.append(self.theta, th)
            self.posX = np.append(self.posX, x)
            self.posY = np.append(self.posY, y)
            self.posX3D = np.append(self.posX3D, x3)
            self.posY3D = np.append(self.posY3D, y3)
            self.posZ3D = np.append(self.posZ3D, z3)
            self.theta_scat = np.append(self.theta_scat, th_sc)
            self.density = np.append(self.density, d)

            ph += self.phase_step
            ph = ph - int(ph)
            dth = 2 * self.area * self.phase_step / (rad * rad)
            th += dth
            rad = (self.a * (1 - self.eccentricity**2) /
                   (1 + self.eccentricity * math.cos(th + math.pi)))
            x, y = rad * math.cos(th + self.omega), rad * math.sin(th + self.omega)
            x3, y3, z3 = x, y * math.cos(self.inclination), y * math.sin(self.inclination)
            rad_vec = np.array([x3, y3, z3])
            th_sc = arccos(np.dot(rad_vec, obs_vec) / norm(rad_vec))
            d = (d_norm / (rad * u.au)**2).to(u.erg / u.cm**3).value

    def set_points(self, mjd_pts=None, phase=None):
        if self.period is None or self.mjd_0 is None:
            print('Cannot set points - set period and/or mjd_per')
            return

        self.mjd_pts, self.phase_pts = np.array([]), np.array([])
        self.radius_pts, self.posX_pts, self.posY_pts = np.array([]), np.array([]), np.array([])
        self.theta_scat_pts, self.density_pts = np.array([]), np.array([])
        self.pos3D_pts = list()

        if mjd_pts is not None:
            self.mjd_pts = [mjd_pts] if not isinstance(mjd_pts, list) else mjd_pts
            self.n_pts = len(self.mjd_pts)
            for m in self.mjd_pts:
                ph = (m - self.mjd_0) / self.period
                ph -= int(ph)
                self.phase_pts = np.append(self.phase_pts, ph)
        elif phase is not None:
            self.phase_pts = [phase] if not isinstance(phase, list) else phase
            self.n_pts = len(self.phase_pts)
        else:
            return

        for ph in self.phase_pts:
            idx = np.abs(np.array(self.phase) - ph).argmin()
            self.radius_pts = np.append(self.radius_pts, self.radius[idx])
            self.posX_pts = np.append(self.posX_pts, self.posX[idx])
            self.posY_pts = np.append(self.posY_pts, self.posY[idx])
            self.theta_scat_pts = np.append(self.theta_scat_pts, self.theta_scat[idx])
            self.density_pts = np.append(self.density_pts, self.density[idx])
            pos3D = np.array([self.posX3D[idx], self.posY3D[idx], self.posZ3D[idx]])
            self.pos3D_pts.append(pos3D)


def generate_systems(eccentricity, phase_per,
                     inclination, omega,
                     period, mjd_0,
                     temp_star, rad_star,
                     mass_star, mass_compact,
                     f_m, x1):
    systems = list()
    comb = itertools.product(*[eccentricity, phase_per,
                               inclination, omega,
                               period, mjd_0,
                               temp_star, rad_star,
                               mass_star, mass_compact, f_m, x1])
    is_ref = True
    for c in comb:
        ecc, ph, inc, om, per, mjd_0, temp, rad, m1, m2, f, x1 = c
        sys = SystemParameters(eccentricity=ecc,
                               phase_per=ph,
                               inclination=inc,
                               omega=om,
                               period=per,
                               mjd_0=mjd_0,
                               temp_star=temp,
                               rad_star=rad,
                               mass_star=m1,
                               mass_compact=m2,
                               f_m=f,
                               x1=x1, is_ref=is_ref)
        systems.append(sys)
        is_ref = False

    return systems


if __name__ == '__main__':

    # systems_ca = generate_systems(eccentricity=[0.83, 0.75, 0.91],
    #                               phase_per=[0.967],
    #                               inclination=[63.7 * degToRad, 45 * degToRad, 80 * degToRad],
    #                               omega=[129 * degToRad, 112 * degToRad, 136 * degToRad],
    #                               period=[321],
    #                               mjd_0=[54857.5],
    #                               temp_star=[30e3],
    #                               rad_star=[7.8],
    #                               mass_star=[16],
    #                               mass_compact=[1.4],
    #                               x1=[0.362, 0.101, 0.623])

    # systems_mo = generate_systems(eccentricity=[0.64, 0.35, 0.93],
    #                               phase_per=[0.663],
    #                               inclination=[63.7 * degToRad, 45 * degToRad, 80 * degToRad],
    #                               omega=[271 * degToRad, 242 * degToRad, 300 * degToRad],
    #                               period=[313],
    #                               mjd_0=[54857.5],
    #                               temp_star=[30e3],
    #                               rad_star=[7.8],
    #                               mass_star=[16],
    #                               mass_compact=[1.4],
    #                               x1=[0.120, 0.091, 0.149])

    label = 'Disk'
    systems_ca = generate_systems(eccentricity=[0.83],
                                  phase_per=[0.967],
                                  inclination=[63.5 * util.degToRad, 47 * util.degToRad, 80 * util.degToRad],
                                  omega=[129 * util.degToRad],
                                  period=[315],
                                  mjd_0=[54857.5],
                                  temp_star=[30e3],
                                  rad_star=[7.8],
                                  mass_star=[16],
                                  mass_compact=[1.5],
                                  f_m=[0.01],
                                  x1=[0.362])

    systems_mo = generate_systems(eccentricity=[0.64],
                                  phase_per=[0.663],
                                  inclination=[35 * util.degToRad, 27 * util.degToRad, 44 * util.degToRad],
                                  omega=[271 * util.degToRad],
                                  period=[313],
                                  mjd_0=[54857.5],
                                  temp_star=[30e3],
                                  rad_star=[7.8],
                                  mass_star=[16],
                                  mass_compact=[1.5],
                                  f_m=[0.0024],
                                  x1=[0.120])
    mjd_pts = [58079, 58101]
    orbits_ca = SetOfOrbits(phase_step=0.0005, color='r', systems=systems_ca, mjd_pts=mjd_pts)
    orbits_mo = SetOfOrbits(phase_step=0.0005, color='b', systems=systems_mo, mjd_pts=mjd_pts)

    orbits_ca.print_pts()
    orbits_mo.print_pts()

    plt.figure(figsize=(20, 14), tight_layout=True)
    plt.subplot(2, 2, 1)
    ax = plt.gca()
    orbits_ca.plot_orbit(noPoint=False)
    orbits_mo.plot_orbit(noPoint=False)

    d_alpha_mo = plt.Circle((0, 0), 1.12,
                            color='black', fill=False, lw=2, ls=':')
    d_gamma_mo = plt.Circle((0, 0), 0.26,
                            color='black', fill=False, lw=2, ls=':')

    d_alpha_za = plt.Circle((0, 0), 0.93,
                            color='black', fill=False, lw=2, ls='--')
    d_gamma_za = plt.Circle((0, 0), 0.23,
                            color='black', fill=False, lw=2, ls='--')

    ax.add_artist(d_alpha_mo)
    ax.add_artist(d_gamma_mo)
    ax.add_artist(d_alpha_za)
    ax.add_artist(d_gamma_za)

    plt.subplot(2, 2, 2)
    ax = plt.gca()
    orbits_ca.plot_distance(noPoint=False)
    orbits_mo.plot_distance(noPoint=False)

    plt.subplot(2, 2, 3)
    ax = plt.gca()
    orbits_ca.plot_theta_scat(noPoint=False)
    orbits_mo.plot_theta_scat(noPoint=False)

    plt.subplot(2, 2, 4)
    ax = plt.gca()
    orbits_ca.plot_density(noPoint=False)
    orbits_mo.plot_density(noPoint=False)

    plt.savefig('figures/Orbits' + label + '.pdf', format='pdf')
    plt.savefig('figures/Orbits' + label + '.png', format='png')
    plt.show()
