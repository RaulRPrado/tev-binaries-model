#!/usr/bin/python3

import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import itertools
import logging
import math
import json
from scipy import stats
from numpy.linalg import norm
from pathlib import Path

import astropy.units as u
import astropy.constants as const
from iminuit import Minuit
from naima.models import (
    ExponentialCutoffPowerLaw,
    PowerLaw,
    Synchrotron,
    InverseCompton
)

import tgblib.pulsar as psr
import tgblib.radiative as rad
import tgblib.absorption as abso
import tgblib.parameters as pars
from tgblib.data import get_data_ls5039
from tgblib import util
from tgblib import orbit

logging.getLogger().setLevel(logging.INFO)


def getSuzakuFlux(sed, energy):
    flux = 0
    for ii in range(len(sed) - 1):
        en = 0.5 * (energy[ii] + energy[ii + 1])
        delta = energy[ii + 1] - energy[ii]
        mean = (sed[ii] + sed[ii + 1]) / en
        flux += delta * mean
    return flux


def getHESSFluxAndGamma(sed, energy):
    sedUnit = [(s / ((e * u.keV)**2)).to(1/(u.TeV * u.cm * u.cm * u.s)) for (s, e) in zip(sed, energy)]

    nSum, gSum = 0, 0
    for ii in range(len(energy) - 1):
        g = - math.log(sedUnit[ii]/sedUnit[ii+1]) / math.log(energy[ii]/energy[ii+1])
        n = sedUnit[ii] * math.pow(energy[ii] * u.keV.to(u.TeV), -g)
        nSum += n
        gSum += g
    nMean = nSum / (len(energy) - 1)
    gMean = gSum / (len(energy) - 1)
    return nMean, gMean


def plotResults(**kwargs):
    # Plotting
    if 'phaseData' not in kwargs.keys():
        logging.error('No phaseData to plot')
        return

    # Norm
    # if 'nFit' in kwargs.keys():
    #     plt.figure(figsize=(8, 6))
    #     ax = plt.gca()
    #     ax.plot(kwargs['phaseData'], kwargs['nFit'], color='r', linestyle='none', marker='*')
    #     plt.show()

    # Flux SUZAKU
    if 'fluxSuzaku' in kwargs.keys() and 'fluxFitSuzaku' in kwargs.keys():
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.plot(
            kwargs['phaseData'],
            [f.value * 1e12 for f in kwargs['fluxFitSuzaku']],
            color='r',
            linestyle='none',
            marker='*'
        )
        ax.errorbar(
            kwargs['phaseData'],
            kwargs['fluxSuzaku'],
            yerr=kwargs['fluxErrSuzaku'],
            color='k',
            linestyle='none',
            marker='o',
            fillstyle='none'
        )
        plt.show()

    # Flux HESS
    if 'fluxHESS' in kwargs.keys() and 'fluxFitHESS' in kwargs.keys():
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.plot(
            kwargs['phaseData'],
            [f.value * 1e12 for f in kwargs['fluxFitHESS']],
            color='r',
            linestyle='none',
            marker='*'
        )
        ax.errorbar(
            kwargs['phaseData'],
            kwargs['fluxHESS'],
            yerr=kwargs['fluxErrHESS'],
            color='k',
            linestyle='none',
            marker='o',
            fillstyle='none'
        )
        plt.show()

    # Gamma HESS
    if 'gammaHESS' in kwargs.keys() and 'gammaFitHESS' in kwargs.keys():
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.plot(
            kwargs['phaseData'],
            kwargs['gammaFitHESS'],
            color='r',
            linestyle='none',
            marker='*'
        )
        ax.errorbar(
            kwargs['phaseData'],
            kwargs['gammaHESS'],
            yerr=kwargs['gammaErrHESS'],
            color='k',
            linestyle='none',
            marker='o',
            fillstyle='none'
        )
        plt.show()

    # Plotting SED
    if 'modelAll' in kwargs.keys() and 'energyAll' in kwargs.keys():
        whichPhase = 7
        thisModel = [(kwargs['nFit'][whichPhase] / 1e20) * m for m in kwargs['modelAll'][whichPhase]]

        sedHESS = [f for (f, e) in zip(thisModel, kwargs['energyAll']) if e > 1e3]
        energyHESS = [e for e in kwargs['energyAll'] if e > 1e3]

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(energyHESS, [f.value * 1e12 for f in sedHESS], color='r', linestyle='-', marker='*')
        measuredSED = [e*e*kwargs['fluxHESS'][whichPhase]*(e**(-kwargs['gammaHESS'][whichPhase])) for e in [e * 1e-9 for e in energyHESS]]
        ax.plot(energyHESS, measuredSED, color='b', linestyle='-')
        plt.show()


def do_fit(
    ThetaIC,   # np.array([90, 90]),
    Pos3D,     # [[1, 1, 1], [1, 1, 1]],
    Dist,      # np.array([2, 2]),
    label='',
    do_abs=True,
    lgEdot_min=31,
    lgEdot_max=37.5,
    lgEdot_bins=10,
    lgSigma_min=-3,
    lgSigma_max=-1,
    lgSigma_bins=10,
    Tstar=pars.TSTAR_LS,
    Rstar=pars.RSTAR_LS,
    AlphaSigma=1,
    Mdot=pars.MDOT_LS,
    Vw=pars.VW_LS
):

    logging.info('Starting fitting')
    OutFit = open('fit_results/fit_results_ls5039_' + label + '.txt', 'w')

    # Loading data
    phaseData, fluxSuzaku, fluxErrSuzaku, gammaSuzaku, gammaErrSuzaku = get_data_ls5039('SUZAKU')
    phaseData, fluxHESS, fluxErrHESS, gammaHESS, gammaErrHESS = get_data_ls5039('HESS')

    logging.debug('PhaseData')
    logging.debug(phaseData)

    # Loading energy
    energyXrays = np.logspace(0, 1, 5)
    energyGamma = np.logspace(math.log10(0.2e9), math.log10(5e9), 10)
    energyAll = np.concatenate((energyXrays, energyGamma))
    logging.debug('Energies')
    logging.debug(energyXrays)
    logging.debug(energyGamma)
    logging.debug(energyAll)

    # Loading grid
    NEdot = int((lgEdot_max - lgEdot_min) * lgEdot_bins)
    NSigma = int((lgSigma_max - lgSigma_min) * lgSigma_bins)
    Edot_list = np.logspace(lgEdot_min, lgEdot_max, int(NEdot))
    Sigma_list = np.logspace(lgSigma_min, lgSigma_max, int(NSigma))

    logging.info('{} iterations'.format(len(Edot_list) * len(Sigma_list)))

    if (
        len(ThetaIC) != len(phaseData)
        or len(Pos3D) != len(phaseData)
        or len(Dist) != len(phaseData)
    ):
        logging.error('Argument with wrong dimensions - aborting')
        return None

    # Absorption
    if not do_abs:
        logging.info('Skipping absorption')
    else:
        if Path('abs_pars.json').exists():
            logging.debug('Reading abs pars')
            with open('abs_pars.json', 'r') as file:
                data = json.load(file)

            DistFrac = data['DistFrac']
            DistTau = data['DistTau']
            Tau = data['Tau']
        else:
            # Computing Taus
            logging.info('Computing absorption')
            start = time.time()
            Obs = np.array([0, 0, -1])
            Abs = abso.Absorption(Tstar=Tstar, Rstar=Rstar, name_table='absorption_table_ls5039.dat')

            Tau = list()
            DistTau = list()
            for iph in range(len(phaseData)):
                DistTau.append(list())
                tt = dict()
                for i in range(len(energyAll)):
                    tt[i] = list()
                Tau.append(tt)

            DistFrac = np.concatenate(
                (np.linspace(0.005, 0.2, 20), np.linspace(0.201, 1.0, 10))
            )

            for ff in DistFrac:
                for iph in range(len(phaseData)):
                    dist = ff * norm(Pos3D[iph])
                    PosStar = Pos3D[iph] * dist / norm(Pos3D[iph])
                    for ien in range(len(energyAll)):
                        tau = Abs.TauGG(en=energyAll[ien] * u.keV.to(u.TeV), obs=Obs, pos=PosStar)
                        Tau[iph][ien].append(tau)
                    DistTau[iph].append(dist)

            logging.info('Abs done, dt/s = {}'.format(time.time() - start))

            absToSave = dict()
            absToSave['DistFrac'] = list(DistFrac)
            absToSave['DistTau'] = list(DistTau)
            absToSave['Tau'] = list(Tau)

            logging.debug('Writing abs to json file')
            with open('abs_pars.json', 'w') as file:
                json.dump(absToSave, file)

        # # Ploting tau vs dist
        # plt.figure(figsize=(8, 6), tight_layout=True)
        # ax = plt.gca()
        # ax.set_yscale('log')
        # ax.set_ylabel(r'$\tau_{\gamma \gamma}$')
        # ax.set_xlabel(r'$D$ [AU]')

        # ax.plot(DistTau[7], Tau[7][str(5)], marker='o', linestyle='-')
        # ax.plot(DistTau[7], Tau[7][str(13)], marker='o', linestyle='-')
        # plt.show()

    chisqMin = 1e10
    minEdot = 0
    minSigma = 0

    for (Edot, Sigma) in itertools.product(Edot_list, Sigma_list):

        print('Starting Edot={}, Sigma={}'.format(Edot, Sigma))

        # Computed parameters
        DistPulsar = [psr.Rshock(Edot=Edot, Mdot=Mdot, Vw=Vw, D=d) for d in Dist]
        DistStar = Dist - DistPulsar
        SigmaFac = [pow(0.1 / d, AlphaSigma) for d in DistPulsar]
        SigmaShock = [Sigma * f for f in SigmaFac]
        Bfield = [psr.B2_KC(Edot=Edot, Rs=dp, sigma=s) for (dp, s) in zip(DistPulsar, SigmaShock)]
        Density = [psr.PhotonDensity(Tstar=Tstar, Rstar=Rstar, d=d) for d in DistStar]

        # plt.figure(figsize=(8, 6), tight_layout=True)
        # ax = plt.gca()
        # ax.set_yscale('log')
        # ax.set_ylabel(r'$\tau_{\gamma \gamma}$')
        # ax.set_xlabel(r'$D$ [AU]')

        # ax.plot(DistPulsar, SigmaShock, marker='o', linestyle='-')
        # plt.show()

        # logging.debug('DistPulsar')
        # logging.debug(DistPulsar)
        # logging.debug('DistStar')
        # logging.debug(DistStar)
        # logging.debug('SigmaShock')
        # logging.debug(SigmaShock)
        # logging.debug('SigmaFac')
        # logging.debug(SigmaFac)
        logging.debug('Bfield')
        logging.debug(Bfield)
        logging.debug('MeanBfield')
        logging.debug(sum(Bfield)/len(Bfield))

        if 0 in Bfield:
            logging.info('Bfield is 0 - skipping')
            continue

        # Normalization
        NormStart = np.array([1e23 / b for b in Bfield])

        npar = len(phaseData)

        # Further parameters
        Ecut = [rad.Emax(b, 1) * u.TeV for b in Bfield]
        Eref = 1 * u.TeV
        Emax = 100 * u.PeV
        Emin = 10 * u.GeV
        SourceDist = pars.SRC_DIST_LS * u.kpc

        # Computing Alpha
        Alpha = [2 * g - 1 for g in gammaSuzaku]
        # print('Alpha')
        # print(Alpha)
        # print('gamma')
        # print(gammaSuzaku)

        # Computing Model
        tau = list()
        modelAll = list()
        for iph in range(len(phaseData)):

            thisTau = list()
            if do_abs:
                for ien in range(len(energyAll)):
                    thisTau.append(np.interp(DistStar[iph], xp=DistTau[iph], fp=Tau[iph][str(ien)]))
            else:
                thisTau = [0] * len(energyAll)

            tau.append(thisTau)

            ECPL = ExponentialCutoffPowerLaw(
                amplitude=1e20 / u.eV,
                e_0=Eref,
                alpha=Alpha[iph],
                e_cutoff=Ecut[iph]
            )

            SYN = Synchrotron(
                particle_distribution=ECPL,
                B=Bfield[iph] * u.G,
                Eemax=Emax,
                Eemin=Emin
            )

            IC = InverseCompton(
                particle_distribution=ECPL,
                seed_photon_fields=[[
                    'STAR',
                    Tstar * u.K,
                    Density[iph] * u.erg / u.cm**3,
                    ThetaIC[iph] * u.deg
                ]],
                Eemax=Emax,
                Eemin=Emin
            )

            thisModel = (
                SYN.sed(photon_energy=[e * u.keV for e in energyAll], distance=SourceDist) +
                IC.sed(photon_energy=[e * u.keV for e in energyAll], distance=SourceDist)
            )

            if do_abs:
                thisModel = [math.exp(-t) * m for (m, t) in zip(thisModel, thisTau)]

            # Ploting tau vs dist
            # plt.figure(figsize=(8, 6), tight_layout=True)
            # ax = plt.gca()
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            # ax.set_ylabel(r'$\tau_{\gamma \gamma}$')
            # ax.set_xlabel(r'$E$ [keV]')

            # print(energyAll)
            # print(thisTau)

            # ax.plot(energyAll[5:], thisTau[5:], marker='o', linestyle='-')
            # ax.set_xlim(1e8, 1e10)
            # plt.show()

            modelAll.append(thisModel)
        # END for

        fluxModelSuzaku, fluxModelHESS, gammaModelHESS = list(), list(), list()
        for thisModel in modelAll:
            sedSuzaku = [f for (f, e) in zip(thisModel, energyAll) if e < 1e3]
            energySuzaku = [e for e in energyAll if e < 1e3]
            sedHESS = [f for (f, e) in zip(thisModel, energyAll) if e > 1e3]
            energyHESS = [e for e in energyAll if e > 1e3]
            fluxModelSuzaku.append(getSuzakuFlux(sedSuzaku, energySuzaku))
            n, g = getHESSFluxAndGamma(sedHESS, energyHESS)
            fluxModelHESS.append(n)
            gammaModelHESS.append(g)

        def computeModelPars(N, model, energy):
            thisModel = [(N / 1e20) * m for m in model]

            sedSuzaku = [f for (f, e) in zip(thisModel, energy) if e < 1e3]
            energySuzaku = [e for e in energyAll if e < 1e3]
            sedHESS = [f for (f, e) in zip(thisModel, energy) if e > 1e3]
            energyHESS = [e for e in energyAll if e > 1e3]

            thisFluxModelSuzaku = getSuzakuFlux(sedSuzaku, energySuzaku)
            thisFluxModelHESS, thisGammaModelHESS = getHESSFluxAndGamma(sedHESS, energyHESS)
            return thisFluxModelSuzaku, thisFluxModelHESS, thisGammaModelHESS

        chisqFit, nFit = list(), list()
        fluxFitSuzaku, fluxFitHESS, gammaFitHESS = list(), list(), list()

        for ii in range(len(phaseData)):

            def least_square(n):
                chisq = 0

                fitFluxModelSuzaku, fitFluxModelHESS, fitGammaModelHESS = computeModelPars(n, modelAll[ii], energyAll)

                chisq += ((fitFluxModelSuzaku.value * 1e12 - fluxSuzaku[ii]) / fluxErrSuzaku[ii])**2
                chisq += ((fitFluxModelHESS.value * 1e12 - fluxHESS[ii]) / fluxErrHESS[ii])**2
                # chisq += ((fitGammaModelHESS - gammaHESS[ii]) / gammaErrHESS[ii])**2

                return chisq

            minuit = Minuit(
                least_square,
                n=NormStart[ii],
                fix_n=False,
                limit_n=(NormStart[ii] * 0.0001, NormStart[ii] * 10000),
                errordef=1.
            )

            fmin, param = minuit.migrad()

            chisqFit.append(minuit.fval)
            nFit.append(param[0]['value'])

            fitFS, fitFH, fitGH = computeModelPars(param[0]['value'], modelAll[ii], energyAll)
            fluxFitSuzaku.append(fitFS)
            fluxFitHESS.append(fitFH)
            gammaFitHESS.append(fitGH)

        # print('N*B')
        # print([n * b for (n,b) in zip(nFit, Bfield)])
        # print([c / 2 for c in chisqFit])
        totalChiSq = sum(chisqFit) / 10.

        if totalChiSq < chisqMin:
            chisqMin = totalChiSq
            minEdot = Edot
            minSigma = Sigma

        print('ChiSq = {}'.format(totalChiSq))

        if totalChiSq < 1e20:
            plotResults(
                phaseData=phaseData,
                nFit=nFit,
                fluxFitSuzaku=fluxFitSuzaku,
                fluxSuzaku=fluxSuzaku,
                fluxErrSuzaku=fluxErrSuzaku,
                fluxFitHESS=fluxFitHESS,
                fluxHESS=fluxHESS,
                fluxErrHESS=fluxErrHESS,
                gammaFitHESS=gammaFitHESS,
                gammaHESS=gammaHESS,
                gammaErrHESS=gammaErrHESS,
                modelAll=modelAll,
                energyAll=energyAll
            )

        continue

        # logging.info('Fit')
        # logging.info('ChiSq/ndf = {}'.format(chisq_min / ndf))
        # logging.info('ChiSq - ndf = {}'.format(chisq_min - ndf))

        # # plot testing
        # for ii, nn in enumerate(n_fit):
        #     if fix_n[ii]:
        #         continue
        #     plt.figure(figsize=(8, 6), tight_layout=True)
        #     ax = plt.gca()
        #     ax.set_xscale('log')
        #     ax.set_yscale('log')
        #     ax.set_title(ii)

        #     ax.errorbar(
        #         data_en[ii],
        #         data_fl[ii],
        #         yerr=data_fl_er[ii],
        #         marker='o',
        #         linestyle='none'
        #     )

        #     ax.plot(
        #         data_en[ii],
        #         [(nn / 1e20) * m.value for m in model[ii]],
        #         marker='o',
        #         linestyle='none'
        #     )
        #     plt.show()

    #     print('p-value', p_value)

    #     if do_abs:
    #         TauPrint0 = [tau0[len(data_en0) - 4], tau0[len(data_en0) - 1]]
    #         TauPrint1 = [tau1[len(data_en1) - 3], tau1[len(data_en1) - 1]]
    #     else:
    #         TauPrint0 = [0, 0]
    #         TauPrint1 = [0, 0]

        if chisq_min - ndf < 1e3:
            OutFit.write(str(chisq_min) + ' ')
            OutFit.write(str(ndf) + ' ')
            for ii in range(pars.MAX_PERIOD):
                if not fix_n[ii]:
                    OutFit.write(str(math.log10(n_fit[ii])) + ' ')
            OutFit.write(str(math.log10(Edot)) + ' ')
            OutFit.write(str(math.log10(Sigma)) + ' ')
            for ii in range(pars.MAX_PERIOD):
                if not fix_n[ii]:
                    idx = periods.index(ii)
                    OutFit.write(str(Dist[idx]) + ' ')
                    OutFit.write(str(DistPulsar[idx]) + ' ')
                    OutFit.write(str(Bfield[idx]) + ' ')
    #         OutFit.write(str(TauPrint0[0]) + ' ')
    #         OutFit.write(str(TauPrint0[1]) + ' ')
    #         OutFit.write(str(TauPrint1[0]) + ' ')
    #         OutFit.write(str(TauPrint1[1]) + '\n')
            OutFit.write('\n')

    print('ChiSqMin {}'.format(chisqMin))
    print('lgEdot {}'.format(math.log10(minEdot)))
    print('Sigma {}'.format(minSigma))

    OutFit.close()


def process_labels(labels):

    inPars = dict()
    inPars['nPars'] = len(labels)
    inPars['label'] = list()
    inPars['orb'] = list()
    inPars['inclination'] = list()
    inPars['Mdot'] = list()
    inPars['omega'] = list()
    inPars['eccentricity'] = list()
    inPars['lgEdot_bins'] = list()
    inPars['lgSigma_bins'] = list()
    inPars['AlphaSigma'] = list()
    inPars['NoAbs'] = list()

    for ll in labels:

        # Orbit
        if 'ca' in ll:
            orb = 'ca'
        elif 'ar' in ll:
            orb = 'ar'
        elif 'sa' in ll or 'test' in ll:
            orb = 'sa'
        else:
            logging.error('ParameterError: unidentified orbit - aborting')
            continue

        inPars['label'].append(ll)

        inPars['orb'].append(orb)

        if orb == 'ca':
            inPars['inclination'].append(24.9)
        elif orb == 'sa':
            inPars['inclination'].append(60)

        inPars['Mdot'].append(1.e-7)

        if orb == 'ca':
            inPars['omega'].append(225.8)
        elif orb == 'sa':
            inPars['omega'].append(237.3)

        if orb == 'ca':
            inPars['eccentricity'].append(0.35)
        elif orb == 'sa':
            inPars['eccentricity'].append(0.24)

        # Size
        if 'test' in ll:
            lgEdot_bins = 20
            lgSigma_bins = 20
        elif 'small' in ll:
            lgEdot_bins = 20
            lgSigma_bins = 40
        else:
            lgEdot_bins = 40
            lgSigma_bins = 200
        inPars['lgEdot_bins'].append(lgEdot_bins)
        inPars['lgSigma_bins'].append(lgSigma_bins)

        # Alpha
        inPars['AlphaSigma'].append(1.0)

        # NoAbs
        if 'no_abs' in ll or 'test' in ll:
            NoAbs = False
        else:
            NoAbs = False
        inPars['NoAbs'].append(NoAbs)

    return inPars


if __name__ == '__main__':

    labels = sys.argv[1:] if len(sys.argv) > 1 else ['test']
    logging.info('Labels {}'.format(labels))

    lgEdot_min = 37.5
    lgEdot_max = 38.2
    lgSigma_min = math.log10(1e-2)
    lgSigma_max = math.log10(4e-2)

    inPars = process_labels(labels)

    for iPar in range(inPars['nPars']):

        # Orbits
        if inPars['orb'][iPar] == 'ca':
            systems = orbit.generate_systems(
                eccentricity=[inPars['eccentricity'][iPar]],
                phase_per=[0],
                inclination=[inPars['inclination'][iPar] * util.degToRad],  # 47-80
                omega=[inPars['omega'][iPar] * util.degToRad],
                period=[pars.TPER_LS],
                mjd_0=[1943.09],
                temp_star=[pars.TSTAR_LS],
                rad_star=[pars.RSTAR_LS],
                mass_star=[pars.MSTAR_LS],
                mass_compact=[1.4],
                f_m=[0.0053],
                x1=[0.008464]
            )
        elif inPars['orb'][iPar] == 'sa':
            systems = orbit.generate_systems(
                eccentricity=[inPars['eccentricity'][iPar]],
                phase_per=[0],
                inclination=[inPars['inclination'][iPar] * util.degToRad],  # 47-80
                omega=[inPars['omega'][iPar] * util.degToRad],
                period=[pars.TPER_LS],
                mjd_0=[5017.08],
                temp_star=[pars.TSTAR_LS],
                rad_star=[pars.RSTAR_LS],
                mass_star=[pars.MSTAR_LS],
                mass_compact=[1.8],
                f_m=[0.0049],
                x1=[0.008231]
            )
        else:
            logging.error('Wrong orbit')

        orbits = orbit.SetOfOrbits(
            phase_step=0.0005,
            color='r',
            systems=systems,
            phases=np.linspace(0.05, 0.95, 10)
        )

        pts_orbit = orbits.get_pts()
        dist_orbit = pts_orbit['distance']
        theta_orbit = pts_orbit['theta_ic']
        pos_orbit = pts_orbit['pos_3D']

        #######
        # Fit
        do_fit(
            label=inPars['label'][iPar],
            lgEdot_min=lgEdot_min,
            lgEdot_max=lgEdot_max,
            lgEdot_bins=inPars['lgEdot_bins'][iPar],
            lgSigma_min=lgSigma_min,
            lgSigma_max=lgSigma_max,
            lgSigma_bins=inPars['lgSigma_bins'][iPar],
            Dist=np.array(dist_orbit),
            ThetaIC=np.array(theta_orbit),
            Pos3D=pos_orbit,
            Vw=pars.VW_LS,
            AlphaSigma=inPars['AlphaSigma'][iPar],
            Mdot=inPars['Mdot'][iPar],
            do_abs=(not inPars['NoAbs'][iPar])
        )
