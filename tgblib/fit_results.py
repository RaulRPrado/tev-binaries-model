#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from numpy.linalg import norm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.interpolate import make_interp_spline, BSpline
from scipy import interpolate

import naima
import astropy.units as u
import astropy.constants as const
from iminuit import Minuit
from naima.models import (
    ExponentialCutoffPowerLaw,
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton
)

import tgblib.pulsar as psr
import tgblib.radiative as rad
from tgblib import data
from tgblib import absorption


class FitResult(object):
    def __init__(self, label='', color='k', SigmaMax=None, EdotMin=None):
        self.color = color
        self.SigmaMax = SigmaMax if SigmaMax is not None else 1e10
        self.EdotMin = EdotMin if EdotMin is not None else 1e10
        data = np.loadtxt('files_fit/fit_results_' + label + '.txt', unpack=True)
        self.chiSq = data[0]
        self.lgNorm0 = data[2]
        self.lgNorm1 = data[3]
        self.lgEdot = data[4]
        self.lgSigma = data[5]
        self.distPulsar0 = data[7]
        self.bField0 = data[8]
        self.distPulsar1 = data[10]
        self.bField1 = data[11]

        self.ndf = data[1][0]
        self.dist0 = data[6][0]
        self.dist1 = data[9][0]

        chisq_sel = list()
        for (c, s) in zip(self.chiSq, self.lgSigma):
            chisq_sel.append(c if s < math.log10(self.SigmaMax) else 1e20)
        idxMin = np.argmin(chisq_sel)
        self.chiSqMin = self.chiSq[idxMin]
        self.lgEdotMin = self.lgEdot[idxMin]
        self.lgSigmaMin = self.lgSigma[idxMin]
        self.b0Min = self.bField0[idxMin]
        self.distPulsar0Min = self.distPulsar0[idxMin]
        self.b1Min = self.bField1[idxMin]
        self.distPulsar1Min = self.distPulsar1[idxMin]
        self.norm0Min = 10**self.lgNorm0[idxMin]
        self.norm1Min = 10**self.lgNorm1[idxMin]

        print('ChiSqMin/ndf=', round(self.chiSqMin, 2), '/', self.ndf, '=',
              round(self.chiSqMin/self.ndf, 3))

        self.sigma_1s, self.chiSq_1s = list(), list()
        self.lgEdot_1s, self.lgSigma_1s = list(), list()
        self.distPulsar0_1s, self.distPulsar1_1s = list(), list()
        self.b0_1s, self.b1_1s = list(), list()
        self.lgNorm0_1s, self.lgNorm1_1s = list(), list()

        self.sigma_2s, self.chiSq_2s = list(), list()
        self.lgEdot_2s, self.lgSigma_2s = list(), list()

        for i in range(len(self.chiSq)):
            sig = math.sqrt(self.chiSq[i] - self.chiSqMin) if self.chiSq[i] - self.chiSqMin > 0 else 1e20
            no_disk = (self.dist0 - self.distPulsar0[i] > 1.12 and
                       self.dist1 - self.distPulsar1[i] > 1.12)
            if (sig < math.sqrt(6.18) and
                    self.lgEdot[i] > math.log10(self.EdotMin) and
                    self.lgSigma[i] < math.log10(self.SigmaMax)):

                self.sigma_2s.append(sig)
                self.chiSq_2s.append(self.chiSq[i])
                self.lgEdot_2s.append(self.lgEdot[i])
                self.lgSigma_2s.append(self.lgSigma[i])
                if sig < math.sqrt(2.3):
                    self.sigma_1s.append(sig)
                    self.chiSq_1s.append(self.chiSq[i])
                    self.lgEdot_1s.append(self.lgEdot[i])
                    self.lgSigma_1s.append(self.lgSigma[i])
                    self.distPulsar0_1s.append(self.distPulsar0[i])
                    self.distPulsar1_1s.append(self.distPulsar1[i])
                    self.b0_1s.append(self.bField0[i])
                    self.b1_1s.append(self.bField1[i])
                    self.lgNorm0_1s.append(self.lgNorm0[i])
                    self.lgNorm1_1s.append(self.lgNorm1[i])

        # Determining the line along Edot
        self.lgEdotLine, self.lgSigmaLine = list(), list()
        self.lgSigmaInf, self.lgSigmaSup = list(), list()
        self.distPulsar0Line, self.b0Line = list(), list()
        self.distPulsar1Line, self.b1Line = list(), list()
        self.lgNorm0Line, self.lgNorm1Line = list(), list()
        lgEdotSet = list(set(self.lgEdot_1s))
        for i in range(len(lgEdotSet)):
            colSigma = [s for (s, e) in zip(self.lgSigma_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            colChiSq = [c for (c, e) in zip(self.chiSq_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            colDist0 = [d for (d, e) in zip(self.distPulsar0_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            colLgNorm0 = [n for (n, e) in zip(self.lgNorm0_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            colB0 = [b for (b, e) in zip(self.b0_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            colDist1 = [d for (d, e) in zip(self.distPulsar1_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            colLgNorm1 = [n for (n, e) in zip(self.lgNorm1_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            colB1 = [b for (b, e) in zip(self.b1_1s, self.lgEdot_1s) if e == lgEdotSet[i]]
            idxMin = np.argmin(colChiSq)
            self.lgEdotLine.append(lgEdotSet[i])
            self.lgSigmaLine.append(colSigma[idxMin])
            self.lgSigmaInf.append(min(colSigma))
            self.lgSigmaSup.append(max(colSigma))
            self.distPulsar0Line.append(colDist0[idxMin] / self.dist0)
            self.b0Line.append(colB0[idxMin])
            self.lgNorm0Line.append(colLgNorm0[idxMin])
            self.distPulsar1Line.append(colDist1[idxMin] / self.dist1)
            self.b1Line.append(colB1[idxMin])
            self.lgNorm1Line.append(colLgNorm1[idxMin])

        self.lgEdotLine_2s, self.lgSigmaLine_2s = list(), list()
        self.lgSigmaInf_2s, self.lgSigmaSup_2s = list(), list()
        lgEdotSet = list(set(self.lgEdot_2s))
        for i in range(len(lgEdotSet)):
            colSigma = [s for (s, e) in zip(self.lgSigma_2s, self.lgEdot_2s) if e == lgEdotSet[i]]
            colChiSq = [c for (c, e) in zip(self.chiSq_2s, self.lgEdot_2s) if e == lgEdotSet[i]]
            idxMin = np.argmin(colChiSq)
            self.lgEdotLine_2s.append(lgEdotSet[i])
            self.lgSigmaLine_2s.append(colSigma[idxMin])
            self.lgSigmaInf_2s.append(min(colSigma))
            self.lgSigmaSup_2s.append(max(colSigma))

    def plot_solution(self, band=True, line=True, star=True, ms=35, with_lines=False, no_2s=True,
                      ls='-', line_ls='-', label=None):
        ax = plt.gca()

        if band:
            lgEdotBand_2s, lgSigmaSupBand_2s, lgSigmaInfBand_2s = zip(*sorted(zip(self.lgEdotLine_2s,
                                                                                  self.lgSigmaSup_2s,
                                                                                  self.lgSigmaInf_2s)))
            lgEdotBand_1s, lgSigmaSupBand_1s, lgSigmaInfBand_1s = zip(*sorted(zip(self.lgEdotLine,
                                                                                  self.lgSigmaSup,
                                                                                  self.lgSigmaInf)))
            if with_lines:
                plt.plot([10**l for l in lgEdotBand_1s], [10**l for l in lgSigmaInfBand_1s],
                         marker='None', ls=ls, linewidth=3,
                         c=self.color, alpha=0.7, label=label)
                plt.plot([10**l for l in lgEdotBand_1s], [10**l for l in lgSigmaSupBand_1s],
                         marker='None', ls=ls, linewidth=3,
                         c=self.color, alpha=0.7)
                # Connecting the borders
                plt.plot([10**lgEdotBand_1s[-1], 10**lgEdotBand_1s[-1]],
                         [10**lgSigmaInfBand_1s[-1], 10**lgSigmaSupBand_1s[-1]],
                         marker='None', ls=ls, linewidth=3,
                         c=self.color, alpha=0.7)
                plt.plot([10**lgEdotBand_1s[0], 10**lgEdotBand_1s[0]],
                         [10**lgSigmaInfBand_1s[0], 10**lgSigmaSupBand_1s[0]],
                         marker='None', ls=ls, linewidth=3,
                         c=self.color, alpha=0.7)
                if not no_2s:
                    plt.plot([10**l for l in lgEdotBand_2s], [10**l for l in lgSigmaInfBand_2s],
                             marker='None', ls=ls, linewidth=3,
                             c=self.color, alpha=0.3)
                    plt.plot([10**l for l in lgEdotBand_2s], [10**l for l in lgSigmaSupBand_2s],
                             marker='None', ls=ls, linewidth=3,
                             c=self.color, alpha=0.3)
                    # Connecting the borders
                    plt.plot([10**lgEdotBand_2s[-1], 10**lgEdotBand_2s[-1]],
                             [10**lgSigmaInfBand_2s[-1], 10**lgSigmaSupBand_2s[-1]],
                             marker='None', ls=ls, linewidth=3,
                             c=self.color, alpha=0.3)
                    plt.plot([10**lgEdotBand_2s[0], 10**lgEdotBand_2s[0]],
                             [10**lgSigmaInfBand_2s[0], 10**lgSigmaSupBand_2s[0]],
                             marker='None', ls=ls, linewidth=3,
                             c=self.color, alpha=0.3)
            else:
                plt.fill_between([10**l for l in lgEdotBand_1s],
                                 [10**l for l in lgSigmaInfBand_1s],
                                 [10**l for l in lgSigmaSupBand_1s],
                                 color=self.color, alpha=0.4)
                if not no_2s:
                    plt.fill_between([10**l for l in lgEdotBand_2s],
                                     [10**l for l in lgSigmaInfBand_2s],
                                     [10**l for l in lgSigmaSupBand_2s],
                                     color=self.color, alpha=0.2)
            if line:
                lgEdotSorted, lgSigmaSorted = zip(*sorted(zip(self.lgEdotLine, self.lgSigmaLine)))
                plt.plot([10**l for l in lgEdotSorted], [10**l for l in lgSigmaSorted],
                         marker='None', ls=line_ls, c=self.color)
        else:  # not band
            ax.scatter([10**l for l in self.lgEdot_1s],
                       [10**l for l in self.lgSigma_1s],
                       c=self.sigma_1s, label=label)

        if star:
            plt.plot([10**self.lgEdotMin], [10**self.lgSigmaMin],
                     marker='*', c=self.color, markersize=15)

    def plot_star(self, Edot=1e36, marker='*', ms=15):
        idx = np.argmin(np.array([math.fabs(l - math.log10(Edot)) for l in self.lgEdotLine]))
        Edot_star = 10**self.lgEdotLine[idx]
        sig_star = 10**self.lgSigmaLine[idx]
        plt.plot(Edot_star, sig_star,
                 marker=marker, c=self.color, markersize=ms)

    def plot_sigma(self, line=True, star=True):
        if line:
            lgEdotSorted, lgSigmaSorted = zip(*sorted(zip(self.lgEdotLine,
                                                          self.lgSigmaLine)))

            plt.plot([10**l for l in lgEdotSorted], [10*l for l in lgSigmaSorted],
                     marker='None', ls='--', c=self.color)

        if star:
            plt.plot([10**self.lgEdotMin], [10**self.lgSigmaMin],
                     marker='*', ls='None', c=self.color, markersize=10)

    def plot_sigma_dist(self, line=True, star=True, ls='-', label='None', in_cm=True, lw=1):
        if line:
            distSorted, lgSigmaSorted = zip(*sorted(zip(self.distPulsar0Line,
                                                        self.lgSigmaLine)))

            fac = u.au.to(u.cm) if in_cm else 1
            plt.plot([fac * d * self.dist0 for d in distSorted],
                     [10**l for l in lgSigmaSorted],
                     marker='None', ls=ls, c=self.color, label=label, linewidth=lw)

    def plot_crab_sigma(self, alpha=1, ls='-'):
            def comp_sig_crab(rs, alpha):
                return 3e-3 * pow(3e17 * u.cm.to(u.au) / rs, alpha)

            sig_crab = [comp_sig_crab(rs * self.dist0, alpha) for rs in self.distPulsar0Line]

            lgEdotSorted, sigmaSorted = zip(*sorted(zip(self.lgEdotLine,
                                                        sig_crab)))

            plt.plot([10**l for l in lgEdotSorted], sigmaSorted,
                     marker='None', ls=ls, c=self.color)

    def plot_B(self, line=True, star=True, only_0=True, ls='-', label='None'):
        if line:
            lgEdotSorted, b0Sorted, b1Sorted = zip(*sorted(zip(self.lgEdotLine,
                                                               self.b0Line,
                                                               self.b1Line)))
            plt.plot([10**l for l in lgEdotSorted], b0Sorted,
                     marker='None', ls=ls, c=self.color, label=label)
            if not only_0:
                plt.plot([10**l for l in lgEdotSorted], b1Sorted,
                         marker='None', ls=ls, c=self.color)

        if star:
            plt.plot([10**self.lgEdotMin], [self.b0Min],
                     marker='*', ls='None', c=self.color, markersize=10)

            if not only_0:
                plt.plot([10**self.lgEdotMin], [self.b1Min],
                         marker='*', ls='None', c=self.color, markersize=10)

    def plot_esyn(self, only_0=True, ls='-', label='None'):
        Tstar = 30e3
        density0 = [psr.PhotonDensity(Tstar=Tstar, Rstar=7.8, d=self.dist0 * (1 - r))
                    for r in self.distPulsar0Line]
        density1 = [psr.PhotonDensity(Tstar=Tstar, Rstar=7.8, d=self.dist1 * (1 - r))
                    for r in self.distPulsar1Line]
        Esyn0 = [1e9*rad.Esyn(E=rad.Ebreak(B=b, T=Tstar, U=u), B=b) for (b, u) in
                 zip(self.b0Line, density0)]
        Esyn1 = [1e9*rad.Esyn(E=rad.Ebreak(B=b, T=Tstar, U=u), B=b) for (b, u) in
                 zip(self.b1Line, density1)]

        lgEdotSorted, Esyn0Sorted, Esyn1Sorted = zip(*sorted(zip(self.lgEdotLine,
                                                                 Esyn0,
                                                                 Esyn1)))
        plt.plot([10**l for l in lgEdotSorted], Esyn0Sorted,
                 marker='None', ls=ls, c=self.color, label=label)
        if not only_0:
            plt.plot([10**l for l in lgEdotSorted], Esyn1Sorted,
                     marker='None', ls=ls, c=self.color)

    def plot_ebreak(self, only_0=True, ls='-', label='None'):
        Tstar = 30e3
        density0 = [psr.PhotonDensity(Tstar=Tstar, Rstar=7.8, d=self.dist0 * (1 - r))
                    for r in self.distPulsar0Line]
        density1 = [psr.PhotonDensity(Tstar=Tstar, Rstar=7.8, d=self.dist1 * (1 - r))
                    for r in self.distPulsar1Line]
        Ebr0 = [rad.Ebreak(B=b, T=Tstar, U=u) for (b, u) in zip(self.b0Line, density0)]
        Ebr1 = [rad.Ebreak(B=b, T=Tstar, U=u) for (b, u) in zip(self.b1Line, density1)]

        lgEdotSorted, Ebr0Sorted, Ebr1Sorted = zip(*sorted(zip(self.lgEdotLine, Ebr0, Ebr1)))
        plt.plot([10**l for l in lgEdotSorted], Ebr0Sorted,
                 marker='None', ls=ls, c=self.color, label=label)
        if not only_0:
            plt.plot([10**l for l in lgEdotSorted], Ebr1Sorted,
                     marker='None', ls=ls, c=self.color)

    def plot_density(self, line=True, only_0=True, ls='-', label='None'):
        if line:
            density0 = [psr.PhotonDensity(Tstar=3e4, Rstar=7.8, d=self.dist0 * (1 - r))
                        for r in self.distPulsar0Line]
            density1 = [psr.PhotonDensity(Tstar=3e4, Rstar=7.8, d=self.dist1 * (1 - r))
                        for r in self.distPulsar1Line]

            lgEdotSorted, density0Sorted, density1Sorted = zip(*sorted(zip(self.lgEdotLine,
                                                                           density0,
                                                                           density1)))

            plt.plot([10**l for l in lgEdotSorted], density0Sorted,
                     marker='None', ls=ls, c=self.color, label=label)
            if not only_0:
                plt.plot([10**l for l in lgEdotSorted], density1Sorted,
                         marker='None', ls=ls, c=self.color)

    def plot_dist(self, line=True, star=True, label='None', ls='-', ratio=True):
        fac = 1 if ratio else self.dist0
        if line:
            lgEdotSorted, dist0Sorted, dist1Sorted = zip(*sorted(zip(self.lgEdotLine,
                                                                     self.distPulsar0Line,
                                                                     self.distPulsar1Line)))
            plt.plot([10**l for l in lgEdotSorted], [fac * d for d in dist0Sorted],
                     marker='None', ls=ls, c=self.color, label=label)

        if star:
            plt.plot([10**self.lgEdotMin], [fac * self.distPulsar0Min / self.dist0],
                     marker='*', ls='None', c=self.color, markersize=12)

    def plot_optical_depth(self, line=True, star=True, label='None', ls='-',
                           pos0=np.array([1, 1, 1]), Tstar=30e3, Rstar=7.8):

        if line:
            lgEdotSorted, dist0Sorted = zip(*sorted(zip(self.lgEdotLine,
                                                        self.distPulsar0Line)))
            lgEdotPlot, dist0Plot = list(), list()
            for i in range(len(lgEdotSorted)):
                if i % 5 == 0:
                    lgEdotPlot.append(lgEdotSorted[i])
                    dist0Plot.append(dist0Sorted[i])

            Obs = np.array([0, 0, -1])
            Abs = absorption.Absorption(Tstar=Tstar, Rstar=Rstar)

            tau0 = [Abs.TauGG(en=0.2, obs=Obs, pos=pos0 * self.dist0 * (1 - r) / norm(pos0))
                    for r in dist0Plot]
            tau1 = [Abs.TauGG(en=5.0, obs=Obs, pos=pos0 * self.dist0 * (1 - r) / norm(pos0))
                    for r in dist0Plot]

            tau0Min = Abs.TauGG(en=0.2, obs=Obs,
                                pos=pos0 * (self.dist0 - self.distPulsar0Min) / norm(pos0))
            tau1Min = Abs.TauGG(en=5.0, obs=Obs,
                                pos=pos0 * (self.dist0 - self.distPulsar0Min) / norm(pos0))

            plt.plot([10**l for l in lgEdotPlot], tau0,
                     marker='None', ls=ls, c=self.color, label=label)
            plt.plot([10**l for l in lgEdotPlot], tau1,
                     marker='None', ls=ls, c=self.color, label=label)

            if star:
                plt.plot([10**self.lgEdotMin], [tau0Min],
                         marker='*', ls='None', c=self.color, markersize=12)
                plt.plot([10**self.lgEdotMin], [tau1Min],
                         marker='*', ls='None', c=self.color, markersize=12)

    def plot_norm(self):
        ax = plt.gca()

        ax.scatter([10**l for l in self.lgEdot_1s],
                   [10**l0/10**l1 for (l0, l1) in zip(self.lgNorm0_1s, self.lgNorm1_1s)],
                   color=self.color, marker='o', s=3)

    def plot_sed(self, period=0, best_solution=True, Edot=1e36,
                 theta_ic=90, dist=2, pos=np.array([1, 1, 1]),
                 ls='-', label='None',
                 Tstar=30e3, Rstar=7.8, Mdot=3.16e-9, Vw=1500,
                 emin=0.1, ecut=50, fast=False):
        Alpha = np.array([2.58, 2.16])

        Eref = 1 * u.TeV
        Ecut = ecut * u.TeV
        Emax = 20 * u.PeV
        Emin = emin * u.TeV
        SourceDist = 1.4 * u.kpc
        n_en = 1 if fast else 2

        Obs = np.array([0, 0, -1])
        Abs = absorption.Absorption(Tstar=Tstar, Rstar=Rstar)

        if best_solution:
            b_sed = self.b0Min if period == 0 else self.b1Min
            norm_sed = self.norm0Min if period == 0 else self.norm1Min
            dist_sed = self.distPulsar0Min if period == 0 else self.distPulsar1Min
            dist_star = dist - dist_sed
            density_sed = psr.PhotonDensity(Tstar=Tstar, Rstar=Rstar, d=dist_star)
        else:
            idx = np.argmin(np.array([math.fabs(l - math.log10(Edot)) for l in self.lgEdotLine]))
            b_sed = self.b0Line[idx] if period == 0 else self.b1Line[idx]
            norm_sed = 10**self.lgNorm0Line[idx] if period == 0 else 10**self.lgNorm1Line[idx]
            dist_sed = self.distPulsar0Line[idx] if period == 0 else self.distPulsar1Line[idx]
            dist_star = dist * (1 - dist_sed)
            density_sed = psr.PhotonDensity(Tstar=Tstar, Rstar=Rstar, d=dist_star)

        EnergyToPlot = np.concatenate((np.logspace(-0.5, 8.3, n_en * 200),
                                       np.logspace(8.3, 9.5, n_en * 5))) * u.keV

        ECPL = ExponentialCutoffPowerLaw(amplitude=norm_sed / u.eV,
                                         e_0=Eref,
                                         alpha=Alpha[period],
                                         e_cutoff=Ecut)
        # Testing ECBPL
        # alpha_1 = Alpha[period] - 0.5
        # alpha_2 = Alpha[period]

        # norm_bpl = norm_sed * pow(Eref / Emin, alpha_2 - alpha_1)
        # EminFac = 1e-2
        # ECPL = ExponentialCutoffBrokenPowerLaw(amplitude=norm_bpl / u.eV,
        #                                        e_0=Eref,
        #                                        alpha_1=alpha_1,
        #                                        alpha_2=alpha_2,
        #                                        e_break=Emin,
        #                                        e_cutoff=Ecut)

        SYN = Synchrotron(particle_distribution=ECPL,
                          B=b_sed * u.G,
                          Eemax=Emax,
                          Eemin=Emin)
        IC = InverseCompton(particle_distribution=ECPL,
                            seed_photon_fields=[['STAR',
                                                 Tstar * u.K,
                                                 density_sed * u.erg / u.cm**3,
                                                 theta_ic * u.deg]],
                            Eemax=Emax,
                            Eemin=Emin)

        # tau = [Abs.TauGG(en=e.value * u.keV.to(u.TeV), obs=Obs,
        #                  pos=pos * dist_star / norm(pos)) for e in EnergyToPlot]

        tau = list()
        for e in EnergyToPlot:
            if e.value * u.keV.to(u.TeV) < 1e-2:
                tau.append(0)
            else:
                tau.append(Abs.TauGG(en=e.value * u.keV.to(u.TeV), obs=Obs,
                           pos=pos * dist_star / norm(pos)))

        model = (SYN.sed(photon_energy=EnergyToPlot, distance=SourceDist) +
                 IC.sed(photon_energy=EnergyToPlot, distance=SourceDist))
        model_abs = [math.exp(-t) * m.value for (m, t) in zip(model, tau)]

        ax = plt.gca()
        ax.plot(EnergyToPlot, model_abs, ls=ls, c=self.color, label=label)

        # Integrating spectrum
        spec = [m.value / e.value / u.keV.to(u.erg) for (m, e) in zip(model, EnergyToPlot)]
        en = [e.value * u.keV.to(u.erg) for e in EnergyToPlot]
        L = np.trapz(x=en, y=spec) * 4 * math.pi * (1.4 * u.kpc.to(u.cm))**2
        print('SED Luminosity = ', L, 'ergs/s')
