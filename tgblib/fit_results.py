#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import math
import logging
from numpy.linalg import norm

import astropy.units as u
import astropy.constants as const
from naima.models import (
    ExponentialCutoffPowerLaw,
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton
)

import tgblib.pulsar as psr
import tgblib.radiative as rad
import tgblib.parameters as pars
from tgblib import data
from tgblib import absorption
from tgblib import util


class FitResult(object):
    def __init__(self, n_periods, label='', color='k', SigmaMax=None, EdotMin=None):
        self.color = color
        self.SigmaMax = SigmaMax if SigmaMax is not None else 1e10
        self.EdotMin = EdotMin if EdotMin is not None else 1e10
        self.nPeriods = n_periods
        self.loadData(label)

        chisq_sel = list()
        for (c, s) in zip(self.chiSq, self.lgSigma):
            chisq_sel.append(c if s < math.log10(self.SigmaMax) else 1e20)
        idxMin = np.argmin(chisq_sel)
        self.chiSqMin = self.chiSq[idxMin]
        self.lgEdotMin = self.lgEdot[idxMin]
        self.lgSigmaMin = self.lgSigma[idxMin]

        self.bMin = list()
        self.distPulsarMin = list()
        self.normMin = list()

        for ip in range(self.nPeriods):
            self.bMin.append(self.bField[ip][idxMin])
            self.distPulsarMin.append(self.distPulsar[ip][idxMin])
            self.normMin.append(10**self.lgNorm[ip][idxMin])

        msg = (
            'ChiSqMin/ndf=' + str(round(self.chiSqMin, 2)) + '/' + str(self.ndf)
            + '=' + str(round(self.chiSqMin/self.ndf, 3))
        )
        logging.info(msg)

        self.sigma_1s, self.chiSq_1s = list(), list()
        self.lgEdot_1s, self.lgSigma_1s = list(), list()
        self.distPulsar_1s, self.b_1s, self.lgNorm_1s = list(), list(), list()
        for ip in range(self.nPeriods):
            self.distPulsar_1s.append(list())
            self.b_1s.append(list())
            self.lgNorm_1s.append(list())

        self.sigma_2s, self.chiSq_2s = list(), list()
        self.lgEdot_2s, self.lgSigma_2s = list(), list()

        for ii in range(len(self.chiSq)):
            sig = (
                math.sqrt(self.chiSq[ii] - self.chiSqMin)
                if self.chiSq[ii] - self.chiSqMin > 0 else 1e20
            )
            no_disk = (
                self.dist[0] - self.distPulsar[0][ii] > 1.12
                and self.dist[1] - self.distPulsar[1][ii] > 1.12
            )
            # 2 sigma band
            if (
                sig < math.sqrt(6.18)
                and self.lgEdot[ii] > math.log10(self.EdotMin)
                and self.lgSigma[ii] < math.log10(self.SigmaMax)
            ):
                self.sigma_2s.append(sig)
                self.chiSq_2s.append(self.chiSq[ii])
                self.lgEdot_2s.append(self.lgEdot[ii])
                self.lgSigma_2s.append(self.lgSigma[ii])
                # 1 sigma band
                if sig < math.sqrt(2.3):
                    self.sigma_1s.append(sig)
                    self.chiSq_1s.append(self.chiSq[ii])
                    self.lgEdot_1s.append(self.lgEdot[ii])
                    self.lgSigma_1s.append(self.lgSigma[ii])
                    for ip in range(self.nPeriods):
                        self.distPulsar_1s[ip].append(self.distPulsar[ip][ii])
                        self.b_1s[ip].append(self.bField[ip][ii])
                        self.lgNorm_1s[ip].append(self.lgNorm[ip][ii])

        # Determining the line along Edot
        self.lgEdotLine, self.lgSigmaLine = list(), list()
        self.lgSigmaInf, self.lgSigmaSup = list(), list()
        self.distPulsarLine, self.bLine, self.lgNormLine = list(), list(), list()
        for ip in range(self.nPeriods):
            self.distPulsarLine.append(list())
            self.bLine.append(list())
            self.lgNormLine.append(list())
        lgEdotSet = list(set(self.lgEdot_1s))
        for ii in range(len(lgEdotSet)):
            colSigma = [s for (s, e) in zip(self.lgSigma_1s, self.lgEdot_1s) if e == lgEdotSet[ii]]
            colChiSq = [c for (c, e) in zip(self.chiSq_1s, self.lgEdot_1s) if e == lgEdotSet[ii]]
            idxMin = np.argmin(colChiSq)
            self.lgEdotLine.append(lgEdotSet[ii])
            self.lgSigmaLine.append(colSigma[idxMin])
            self.lgSigmaInf.append(min(colSigma))
            self.lgSigmaSup.append(max(colSigma))

            for ip in range(self.nPeriods):

                colDist, colLgNorm, colB = list(), list(), list()
                for ie in range(len(self.lgEdot_1s)):
                    if self.lgEdot_1s[ie] != lgEdotSet[ii]:
                        continue
                    colDist.append(self.distPulsar_1s[ip][ie])
                    colLgNorm.append(self.lgNorm_1s[ip][ie])
                    colB.append(self.b_1s[ip][ie])

                self.distPulsarLine[ip].append(colDist[idxMin] / self.dist[0])
                self.bLine[ip].append(colB[idxMin])
                self.lgNormLine[ip].append(colLgNorm[idxMin])

        self.lgEdotLine_2s, self.lgSigmaLine_2s = list(), list()
        self.lgSigmaInf_2s, self.lgSigmaSup_2s = list(), list()
        lgEdotSet = list(set(self.lgEdot_2s))
        for ii in range(len(lgEdotSet)):
            colSigma = [s for (s, e) in zip(self.lgSigma_2s, self.lgEdot_2s) if e == lgEdotSet[ii]]
            colChiSq = [c for (c, e) in zip(self.chiSq_2s, self.lgEdot_2s) if e == lgEdotSet[ii]]
            idxMin = np.argmin(colChiSq)
            self.lgEdotLine_2s.append(lgEdotSet[ii])
            self.lgSigmaLine_2s.append(colSigma[idxMin])
            self.lgSigmaInf_2s.append(min(colSigma))
            self.lgSigmaSup_2s.append(max(colSigma))

    def loadData(self, label):
        fileName = 'fit_results/fit_results_' + label + '.txt'
        logging.info('Loading data {}'.format(fileName))
        data = np.loadtxt(fileName, unpack=True)
        self.chiSq = data[0]
        self.ndf = data[1][0]
        self.lgNorm = list()
        for ip in range(self.nPeriods):
            self.lgNorm.append(data[2 + ip])
        self.lgEdot = data[2 + self.nPeriods]
        self.lgSigma = data[3 + self.nPeriods]
        self.dist = list()
        self.distPulsar = list()
        self.bField = list()
        for ip in range(self.nPeriods):
            self.dist.append(data[4 + self.nPeriods + 3*ip][0])
            self.distPulsar.append(data[5 + self.nPeriods + 3*ip])
            self.bField.append(data[6 + self.nPeriods + 3*ip])

    def plot_solution(
        self,
        band=True,
        line=True,
        star=True,
        ms=35,
        with_lines=False,
        no_2s=True,
        ls='-',
        line_ls='-',
        label=None
    ):
        ax = plt.gca()

        if band:
            lgEdotBand_2s, lgSigmaSupBand_2s, lgSigmaInfBand_2s = zip(*sorted(zip(
                self.lgEdotLine_2s,
                self.lgSigmaSup_2s,
                self.lgSigmaInf_2s
            )))
            lgEdotBand_1s, lgSigmaSupBand_1s, lgSigmaInfBand_1s = zip(*sorted(zip(
                self.lgEdotLine,
                self.lgSigmaSup,
                self.lgSigmaInf
            )))

            if with_lines:
                plt.plot(
                    [10**l for l in lgEdotBand_1s],
                    [10**l for l in lgSigmaInfBand_1s],
                    marker='None',
                    ls=ls,
                    linewidth=3,
                    c=self.color,
                    alpha=0.7,
                    label=label
                )
                plt.plot(
                    [10**l for l in lgEdotBand_1s],
                    [10**l for l in lgSigmaSupBand_1s],
                    marker='None',
                    ls=ls,
                    linewidth=3,
                    c=self.color,
                    alpha=0.7
                )
                # Connecting the borders
                plt.plot(
                    [10**lgEdotBand_1s[-1], 10**lgEdotBand_1s[-1]],
                    [10**lgSigmaInfBand_1s[-1], 10**lgSigmaSupBand_1s[-1]],
                    marker='None',
                    ls=ls,
                    linewidth=3,
                    c=self.color,
                    alpha=0.7
                )
                plt.plot(
                    [10**lgEdotBand_1s[0], 10**lgEdotBand_1s[0]],
                    [10**lgSigmaInfBand_1s[0], 10**lgSigmaSupBand_1s[0]],
                    marker='None',
                    ls=ls,
                    linewidth=3,
                    c=self.color,
                    alpha=0.7
                )
                if not no_2s:
                    plt.plot(
                        [10**l for l in lgEdotBand_2s],
                        [10**l for l in lgSigmaInfBand_2s],
                        marker='None',
                        ls=ls,
                        linewidth=3,
                        c=self.color,
                        alpha=0.3
                    )
                    plt.plot(
                        [10**l for l in lgEdotBand_2s],
                        [10**l for l in lgSigmaSupBand_2s],
                        marker='None',
                        ls=ls,
                        linewidth=3,
                        c=self.color,
                        alpha=0.3
                    )
                    # Connecting the borders
                    plt.plot(
                        [10**lgEdotBand_2s[-1], 10**lgEdotBand_2s[-1]],
                        [10**lgSigmaInfBand_2s[-1], 10**lgSigmaSupBand_2s[-1]],
                        marker='None',
                        ls=ls,
                        linewidth=3,
                        c=self.color,
                        alpha=0.3
                    )
                    plt.plot(
                        [10**lgEdotBand_2s[0], 10**lgEdotBand_2s[0]],
                        [10**lgSigmaInfBand_2s[0], 10**lgSigmaSupBand_2s[0]],
                        marker='None',
                        ls=ls,
                        linewidth=3,
                        c=self.color,
                        alpha=0.3
                    )
            else:
                plt.fill_between(
                    [10**l for l in lgEdotBand_1s],
                    [10**l for l in lgSigmaInfBand_1s],
                    [10**l for l in lgSigmaSupBand_1s],
                    color=self.color,
                    alpha=0.4
                )
                if not no_2s:
                    plt.fill_between(
                        [10**l for l in lgEdotBand_2s],
                        [10**l for l in lgSigmaInfBand_2s],
                        [10**l for l in lgSigmaSupBand_2s],
                        color=self.color,
                        alpha=0.2
                    )
            if line:
                lgEdotSorted, lgSigmaSorted = zip(*sorted(zip(self.lgEdotLine, self.lgSigmaLine)))
                plt.plot(
                    [10**l for l in lgEdotSorted],
                    [10**l for l in lgSigmaSorted],
                    marker='None',
                    ls=line_ls,
                    c=self.color
                )
        else:  # not band
            ax.scatter(
                [10**l for l in self.lgEdot_1s],
                [10**l for l in self.lgSigma_1s],
                c=self.sigma_1s, label=label
            )

        if star:
            plt.plot(
                [10**self.lgEdotMin],
                [10**self.lgSigmaMin],
                marker='*',
                c=self.color,
                markersize=15
            )

    def plot_star(self, Edot=1e36, marker='*', ms=15):
        idx = np.argmin(np.array([math.fabs(l - math.log10(Edot)) for l in self.lgEdotLine]))
        Edot_star = 10**self.lgEdotLine[idx]
        sig_star = 10**self.lgSigmaLine[idx]
        plt.plot(
            Edot_star,
            sig_star,
            marker=marker,
            c=self.color,
            markersize=ms
        )

    def plot_sigma(self, line=True, star=True):
        if line:
            lgEdotSorted, lgSigmaSorted = zip(*sorted(zip(
                self.lgEdotLine,
                self.lgSigmaLine
            )))

            plt.plot(
                [10**l for l in lgEdotSorted],
                [10*l for l in lgSigmaSorted],
                marker='None',
                ls='--',
                c=self.color
            )

        if star:
            plt.plot(
                [10**self.lgEdotMin],
                [10**self.lgSigmaMin],
                marker='*',
                ls='None',
                c=self.color,
                markersize=10
            )

    def plot_sigma_dist(self, line=True, star=True, ls='-', label='None', in_cm=True, lw=1):
        if line:
            distSorted, lgSigmaSorted = zip(*sorted(zip(
                self.distPulsar0Line,
                self.lgSigmaLine
            )))

            fac = u.au.to(u.cm) if in_cm else 1
            plt.plot(
                [fac * d * self.dist0 for d in distSorted],
                [10**l for l in lgSigmaSorted],
                marker='None',
                ls=ls,
                c=self.color,
                label=label,
                linewidth=lw
            )

    def plot_crab_sigma(self, alpha=1, ls='-'):

        def comp_sig_crab(rs, alpha):
            return 3e-3 * pow(3e17 * u.cm.to(u.au) / rs, alpha)

        sig_crab = [comp_sig_crab(rs * self.dist[0], alpha) for rs in self.distPulsarLine[0]]

        lgEdotSorted, sigmaSorted = zip(*sorted(zip(
            self.lgEdotLine,
            sig_crab
        )))

        plt.plot(
            [10**l for l in lgEdotSorted],
            sigmaSorted,
            marker='None',
            ls=ls,
            c=self.color
        )

    def plot_B(self, line=True, star=True, iperiod=0, ls='-', label='None'):

        if line:
            lgEdotSorted, bSorted = zip(*sorted(zip(
                self.lgEdotLine,
                self.bLine[iperiod]
            )))
            plt.plot(
                [10**l for l in lgEdotSorted],
                bSorted,
                marker='None',
                ls=ls,
                c=self.color,
                label=label
            )

        if star:
            plt.plot(
                [10**self.lgEdotMin],
                [self.bMin[iperiod]],
                marker='*',
                ls='None',
                c=self.color,
                markersize=10
            )

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

    def plot_density(self, line=True, iperiod=0, ls='-', label='None'):
        if line:
            density = [
                psr.PhotonDensity(Tstar=pars.TSTAR, Rstar=pars.RSTAR, d=self.dist[iperiod]*(1 - r))
                for r in self.distPulsarLine[iperiod]
            ]

            lgEdotSorted, densitySorted = zip(*sorted(zip(
                self.lgEdotLine,
                density
            )))

            plt.plot(
                [10**l for l in lgEdotSorted],
                densitySorted,
                marker='None',
                ls=ls,
                c=self.color,
                label=label
            )

    def plot_dist(self, line=True, star=True, iperiod=0, label='None', ls='-', ratio=True):
        fac = 1 if ratio else self.dist[iperiod]
        if line:
            lgEdotSorted, distSorted = zip(*sorted(zip(
                self.lgEdotLine,
                self.distPulsarLine[iperiod]
            )))
            plt.plot(
                [10**l for l in lgEdotSorted],
                [fac * d for d in distSorted],
                marker='None',
                ls=ls,
                c=self.color,
                label=label
            )

        if star:
            plt.plot(
                [10**self.lgEdotMin],
                [fac * self.distPulsarMin[iperiod] / self.dist[iperiod]],
                marker='*',
                ls='None',
                c=self.color,
                markersize=12
            )

    def plot_optical_depth(
        self,
        pos,
        line=True,
        star=True,
        iperiod=0,
        label='None',
        ls='-',
        Tstar=pars.TSTAR,
        Rstar=pars.RSTAR
    ):

        if line:
            lgEdotSorted, distSorted = zip(*sorted(zip(
                self.lgEdotLine,
                self.distPulsarLine[iperiod]
            )))
            lgEdotPlot, distPlot = list(), list()
            for i in range(len(lgEdotSorted)):
                if i % 5 == 0:
                    lgEdotPlot.append(lgEdotSorted[i])
                    distPlot.append(distSorted[i])

            Obs = np.array([0, 0, -1])
            Abs = absorption.Absorption(Tstar=Tstar, Rstar=Rstar)

            tau = [
                Abs.TauGG(en=0.2, obs=Obs, pos=pos * self.dist[iperiod] * (1 - r) / norm(pos))
                for r in distPlot
            ]

            tauMin = Abs.TauGG(
                en=0.2,
                obs=Obs,
                pos=pos * (self.dist[iperiod] - self.distPulsarMin[iperiod]) / norm(pos)
            )

            plt.plot(
                [10**l for l in lgEdotPlot],
                tau,
                marker='None',
                ls=ls,
                c=self.color,
                label=label
            )

            if star:
                plt.plot(
                    [10**self.lgEdotMin],
                    [tauMin],
                    marker='*',
                    ls='None',
                    c=self.color,
                    markersize=12
                )

    def plot_norm(self):
        ax = plt.gca()

        ax.scatter([10**l for l in self.lgEdot_1s],
                   [10**l0/10**l1 for (l0, l1) in zip(self.lgNorm0_1s, self.lgNorm1_1s)],
                   color=self.color, marker='o', s=3)

    def plot_sed(
        self,
        iperiod=0,
        period=0,
        best_solution=True,
        Edot=1e36,
        theta_ic=90,
        dist=2,
        pos=np.array([1, 1, 1]),
        ls='-',
        label='None',
        Tstar=pars.TSTAR,
        Rstar=pars.RSTAR,
        Mdot=3.16e-9,
        Vw=1500,
        emin=0.1,
        ecut=50,
        fast=False
    ):
        Alpha = pars.ELEC_SPEC_INDEX[period]

        Eref = 1 * u.TeV
        Ecut = ecut * u.TeV
        Emax = 20 * u.PeV
        Emin = emin * u.TeV
        SourceDist = pars.SRC_DIST * u.kpc
        n_en = 1 if fast else 2

        Obs = np.array([0, 0, -1])
        Abs = absorption.Absorption(Tstar=Tstar, Rstar=Rstar)

        if best_solution:
            b_sed = self.bMin[iperiod]
            norm_sed = self.normMin[iperiod]
            dist_sed = self.distPulsarMin[iperiod]
            dist_star = dist - dist_sed
            density_sed = psr.PhotonDensity(Tstar=Tstar, Rstar=Rstar, d=dist_star)
        else:
            idx = np.argmin(np.array([math.fabs(l - math.log10(Edot)) for l in self.lgEdotLine]))
            b_sed = self.bLine[iperiod][idx]
            norm_sed = 10**self.lgNormLine[iperiod][idx]
            dist_sed = self.distPulsarLine[iperiod][idx]
            dist_star = dist * (1 - dist_sed)
            density_sed = psr.PhotonDensity(Tstar=Tstar, Rstar=Rstar, d=dist_star)

        EnergyToPlot = np.logspace(-0.5, 9.6, n_en * 200) * u.keV

        ECPL = ExponentialCutoffPowerLaw(
            amplitude=norm_sed / u.eV,
            e_0=Eref,
            alpha=Alpha,
            e_cutoff=Ecut
        )

        SYN = Synchrotron(
            particle_distribution=ECPL,
            B=b_sed * u.G,
            Eemax=Emax,
            Eemin=Emin
        )
        IC = InverseCompton(
            particle_distribution=ECPL,
            seed_photon_fields=[[
                'STAR',
                Tstar * u.K,
                density_sed * u.erg / u.cm**3,
                theta_ic * u.deg
            ]],
            Eemax=Emax,
            Eemin=Emin
        )

        tau = list()
        for e in EnergyToPlot:
            if e.value * u.keV.to(u.TeV) < 1e-2:
                tau.append(0)
            else:
                tau.append(Abs.TauGG(
                    en=e.value * u.keV.to(u.TeV),
                    obs=Obs,
                    pos=pos * dist_star / norm(pos)
                ))

        model = (
            SYN.sed(photon_energy=EnergyToPlot, distance=SourceDist)
            + IC.sed(photon_energy=EnergyToPlot, distance=SourceDist)
        )
        model_abs = [math.exp(-t) * m.value for (m, t) in zip(model, tau)]

        EnergyToPlot, model_abs = util.fix_naima_bug(EnergyToPlot, model_abs)

        ax = plt.gca()
        ax.plot(EnergyToPlot, model_abs, ls=ls, c=self.color, label=label)

        # Integrating spectrum
        # spec = [m.value / e.value / u.keV.to(u.erg) for (m, e) in zip(model, EnergyToPlot)]
        # en = [e.value * u.keV.to(u.erg) for e in EnergyToPlot]
        # L = np.trapz(x=en, y=spec) * 4 * math.pi * (SourceDist * u.kpc.to(u.cm))**2
        # print('SED Luminosity = ', L, 'ergs/s')
