#!/usr/bin/python3

import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import math
import itertools
import logging
from scipy import stats
from numpy.linalg import norm

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
import tgblib.absorption as abso
import tgblib.parameters as pars
from tgblib import data
from tgblib import util
from tgblib import orbit

logging.getLogger().setLevel(logging.DEBUG)


def do_fit(
    periods,  # [0, 1],
    ThetaIC,  # np.array([90, 90]),
    Pos3D,    # [[1, 1, 1], [1, 1, 1]],
    Dist,     # np.array([2, 2]),
    Alpha,     # np.array([2.58, 2.16])
    label='',
    do_abs=True,
    lgEdot_min=31,
    lgEdot_max=37.5,
    lgEdot_bins=10,
    lgSigma_min=-3,
    lgSigma_max=-1,
    lgSigma_bins=10,
    Tstar=pars.TSTAR,
    Rstar=pars.RSTAR,
    AlphaSigma=1,
    Mdot=1e-8,
    Vw=1500
):

    logging.info('Starting fitting')
    OutFit = open('fit_results/fit_results_' + label + '.txt', 'w')

    NEdot = (lgEdot_max - lgEdot_min) * lgEdot_bins
    NSigma = (lgSigma_max - lgSigma_min) * lgSigma_bins
    Edot_list = np.logspace(lgEdot_min, lgEdot_max, int(NEdot))
    Sigma_list = np.logspace(lgSigma_min, lgSigma_max, int(NSigma))

    logging.info('{} iterations'.format(NEdot * NSigma))

    n_periods = len(periods)

    if len(ThetaIC) != n_periods:
        logging.error('Argument with wrong dimensions - aborting')
        return None

    # Absorption
    if do_abs:
        # Computing Taus
        logging.info('Computing absorption')
        start = time.time()
        Obs = np.array([0, 0, -1])
        Abs = abso.Absorption(Tstar=Tstar, Rstar=Rstar)

        data_en, data_fl, data_fl_er = list(), list(), list()
        Tau = list()
        DistTau = list()
        for iper in periods:
            en, fl, fl_er = data.get_data(iper)
            data_en.append(en)
            data_fl.append(fl)
            data_fl_er.append(fl_er)
            DistTau.append(list())
            tt = dict()
            for i in range(len(en)):
                tt[i] = list()
            Tau.append(tt)

        DistFrac = np.linspace(0.01, 1, 50)

        for ff in DistFrac:

            for i_per in range(n_periods):

                dist = ff * norm(Pos3D[i_per])
                PosStar = Pos3D[i_per] * dist / norm(Pos3D[i_per])
                for i_en in range(len(data_en[i_per])):
                    tau = Abs.TauGG(en=data_en[i_per][i_en] * u.keV.to(u.TeV), obs=Obs, pos=PosStar)
                    Tau[i_per][i_en].append(tau)
                DistTau[i_per].append(dist)

        logging.info('Abs done, dt/s = {}'.format(time.time() - start))

        print(Tau)

        # # Ploting tau vs dist
        # plt.figure(figsize=(8, 6), tight_layout=True)
        # ax = plt.gca()
        # ax.set_yscale('log')
        # ax.set_ylabel(r'$\tau_{\gamma \gamma}$')
        # ax.set_xlabel(r'$D$ [AU]')

        # ax.plot(DistTau0, Tau0[92], marker='o', linestyle='-')
        # ax.plot(DistTau0, Tau0[95], marker='o', linestyle='-')
        # plt.show()

    return None

    for (Edot, Sigma) in itertools.product(Edot_list, Sigma_list):

        # Computed parameters
        DistPulsar = [psr.Rshock(Edot=Edot, Mdot=Mdot, Vw=Vw, D=d) for d in Dist]
        DistStar = Dist - DistPulsar
        SigmaFac = pow(Dist[0] / Dist[1], AlphaSigma)
        SigmaShock = [Sigma, Sigma * SigmaFac]
        Bfield = [psr.B2_KC(Edot=Edot, Rs=dp, sigma=s) for (dp, s) in zip(DistPulsar, SigmaShock)]
        Density = [psr.PhotonDensity(Tstar=Tstar, Rstar=Rstar, d=d) for d in DistStar]

        if Bfield[0] == 0 or Bfield[1] == 0:
            continue

        # Normalization
        norm0_start = 1e24 / Bfield[0]
        norm1_start = 1e24 / Bfield[1]

        Norm0 = np.array([norm0_start, norm1_start])

        # Fitting
        fix_n0 = False
        fit_n0 = Norm0[0]

        fix_n1 = False
        fit_n1 = Norm0[1]

        npar = 2 - int(fix_n0) - int(fix_n1)

        ####################
        # Further parameters
        Eref = 1 * u.TeV
        Ecut = 50 * u.TeV
        Emax = 20 * u.PeV
        Emin = 10 * u.GeV
        SourceDist = 1.4 * u.kpc
        EnergyToPlot = np.logspace(-2, 11, 500) * u.keV

        ######################
        # Loading data
        data_en0, data_fl0, data_fl_er0 = data.get_data(0)
        data_en1, data_fl1, data_fl_er1 = data.get_data(1)
        fermi_spec_en, fermi_spec_fl, fermi_spec_fl_er = data.get_fermi_spec()
        fermi_lim_en, fermi_lim_fl, fermi_lim_fl_er = data.get_fermi_upper_limits()

        # Absorption
        if do_abs:
            tau0, tau1 = list(), list()
            for i in range(len(data_en0)):
                t = np.interp(DistStar[0], xp=DistTau0, fp=Tau0[i])
                tau0.append(t)
            for i in range(len(data_en1)):
                t = np.interp(DistStar[1], xp=DistTau1, fp=Tau1[i])
                tau1.append(t)
        else:
            tau0 = [0] * len(data_en0)
            tau1 = [0] * len(data_en1)

        ECPL0 = ExponentialCutoffPowerLaw(amplitude=1e20 / u.eV,
                                          e_0=Eref,
                                          alpha=Alpha[0],
                                          e_cutoff=Ecut,
                                          beta=1)

        SYN0 = Synchrotron(particle_distribution=ECPL0,
                           B=Bfield[0] * u.G,
                           Eemax=Emax,
                           Eemin=Emin)
        IC0 = InverseCompton(particle_distribution=ECPL0,
                             seed_photon_fields=[['STAR',
                                                  Tstar * u.K,
                                                  Density[0] * u.erg / u.cm**3,
                                                  ThetaIC[0] * u.deg]],
                             Eemax=Emax,
                             Eemin=Emin)
        model0 = (SYN0.sed(photon_energy=[e * u.keV for e in data_en0], distance=SourceDist) +
                  IC0.sed(photon_energy=[e * u.keV for e in data_en0], distance=SourceDist))

        ECPL1 = ExponentialCutoffPowerLaw(amplitude=1e20 / u.eV,
                                          e_0=Eref,
                                          alpha=Alpha[1],
                                          e_cutoff=Ecut,
                                          beta=1)

        SYN1 = Synchrotron(particle_distribution=ECPL1,
                           B=Bfield[1] * u.G,
                           Eemax=Emax,
                           Eemin=Emin)
        IC1 = InverseCompton(particle_distribution=ECPL1,
                             seed_photon_fields=[['STAR',
                                                  Tstar * u.K,
                                                  Density[1] * u.erg / u.cm**3,
                                                  ThetaIC[1] * u.deg]],
                             Eemax=Emax,
                             Eemin=Emin)
        model1 = (SYN1.sed(photon_energy=[e * u.keV for e in data_en1], distance=SourceDist) +
                  IC1.sed(photon_energy=[e * u.keV for e in data_en1], distance=SourceDist))

        if do_abs:
            model0 = [math.exp(-t) * m for (m, t) in zip(model0, tau0)]
            model1 = [math.exp(-t) * m for (m, t) in zip(model1, tau1)]

        def least_square(n0, n1):
            fac0 = n0 / 1e20
            fac1 = n1 / 1e20
            chisq = 0
            chisq += sum(util.vecChiSq([fac0 * m.value for m in model0], data_fl0, data_fl_er0))
            chisq += sum(util.vecChiSq([fac1 * m.value for m in model1], data_fl1, data_fl_er1))
            return chisq

        minuit = Minuit(least_square,
                        n0=fit_n0, fix_n0=fix_n0,
                        n1=fit_n1, fix_n1=fix_n1,
                        limit_n0=(norm0_start * 0.001, norm0_start * 1000),
                        limit_n1=(norm1_start * 0.001, norm1_start * 1000))

        fmin, param = minuit.migrad()

        print(minuit.matrix(correlation=True))
        chisq_min = minuit.fval
        n0_fit = param[0]['value']
        n0_err = param[0]['error']
        n1_fit = param[1]['value']
        n1_err = param[1]['error']

        ndf = len(data_en0) + len(data_en1) - npar
        p_value = 1 - stats.chi2.cdf(chisq_min, ndf)

        print('Fit')
        print('ChiSq', chisq_min / ndf)
        print('p-value', p_value)

        if do_abs:
            TauPrint0 = [tau0[len(data_en0) - 4], tau0[len(data_en0) - 1]]
            TauPrint1 = [tau1[len(data_en1) - 3], tau1[len(data_en1) - 1]]
        else:
            TauPrint0 = [0, 0]
            TauPrint1 = [0, 0]

        if chisq_min - ndf < 16:
            OutFit.write(str(chisq_min) + ' ')
            OutFit.write(str(ndf) + ' ')
            OutFit.write(str(math.log10(n0_fit)) + ' ')
            OutFit.write(str(math.log10(n1_fit)) + ' ')
            OutFit.write(str(math.log10(Edot)) + ' ')
            OutFit.write(str(math.log10(Sigma)) + ' ')
            OutFit.write(str(Dist[0]) + ' ')
            OutFit.write(str(DistPulsar[0]) + ' ')
            OutFit.write(str(Bfield[0]) + ' ')
            OutFit.write(str(Dist[1]) + ' ')
            OutFit.write(str(DistPulsar[1]) + ' ')
            OutFit.write(str(Bfield[1]) + ' ')
            OutFit.write(str(TauPrint0[0]) + ' ')
            OutFit.write(str(TauPrint0[1]) + ' ')
            OutFit.write(str(TauPrint1[0]) + ' ')
            OutFit.write(str(TauPrint1[1]) + '\n')

    OutFit.close()


def process_labels(labels):

    inPars = dict()
    inPars['nPars'] = len(labels)
    inPars['label'] = list()
    inPars['orb'] = list()
    inPars['inclination'] = list()
    inPars['Mdot'] = list()
    inPars['period'] = list()
    inPars['omega'] = list()
    inPars['eccentricity'] = list()
    inPars['lgEdot_bins'] = list()
    inPars['lgSigma_bins'] = list()
    inPars['AlphaSigma'] = list()

    for ll in labels:

        # Orbit
        if 'ca' in ll:
            orb = 'ca'
        elif 'mo' in ll:
            orb = 'mo'
        else:
            logging.error('ParameterError: unidentified orbit - aborting')
            continue
        inPars['label'].append(ll)
        inPars['orb'].append(orb)

        # Inclination
        if 'i_inf' in ll:
            inclination = 59 if orb == 'ca' else 32
        elif 'i_sup' in ll:
            inclination = 80 if orb == 'ca' else 42
        else:
            inclination = 69.5 if orb == 'ca' else 37
        inPars['inclination'].append(inclination)

        # Mdot
        if 'm_inf' in ll:
            Mdot = 1e-9
        elif 'm_sup' in ll:
            Mdot = 1e-8
        else:
            Mdot = 3.16e-9
        inPars['Mdot'].append(Mdot)

        # Period
        if 'p_inf' in ll:
            period = 313
        elif 'p_sup' in ll:
            period = 317
        else:
            period = 315
        inPars['period'].append(period)

        # Omega
        if 'o_inf' in ll:
            omega = 112 if orb == 'ca' else 242
        elif 'o_sup' in ll:
            omega = 146 if orb == 'ca' else 300
        else:
            omega = 129 if orb == 'ca' else 271
        inPars['omega'].append(omega)

        # Eccentricity
        if 'e_inf' in ll:
            eccentricity = 0.75 if orb == 'ca' else 0.35
        elif 'e_sup' in ll:
            eccentricity = 0.91 if orb == 'ca' else 0.93
        else:
            eccentricity = 0.83 if orb == 'ca' else 0.64
        inPars['eccentricity'].append(eccentricity)

        # Size
        if 'small' in ll:
            lgEdot_bins = 20
            lgSigma_bins = 40
        else:
            lgEdot_bins = 40
            lgSigma_bins = 200
        inPars['lgEdot_bins'].append(lgEdot_bins)
        inPars['lgSigma_bins'].append(lgSigma_bins)

        # Alpha
        if 'a_inf' in ll:
            AlphaSigma = 0.5
        elif 'a_sup' in ll:
            AlphaSigma = 1.5
        else:
            AlphaSigma = 1.0
        inPars['AlphaSigma'].append(AlphaSigma)

    return inPars


if __name__ == '__main__':

    labels = sys.argv[1:] if len(sys.argv) > 1 else ['ca_small']
    logging.info('Labels {}'.format(labels))

    lgEdot_min = 33
    lgEdot_max = 39
    lgSigma_min = math.log10(1e-3)
    lgSigma_max = math.log10(3e-1)

    periods = [0, 1]
    mjd_pts = [pars.MJD_MEAN[p] for p in periods]
    alphas = [pars.ELEC_SPEC_INDEX[p] for p in periods]

    inPars = process_labels(labels)

    for iPar in range(inPars['nPars']):

        # Orbits
        if inPars['orb'][iPar] == 'ca':
            systems = orbit.generate_systems(
                eccentricity=[inPars['eccentricity'][iPar]],
                phase_per=[0.967],
                inclination=[inPars['inclination'][iPar] * util.degToRad],  # 47-80
                omega=[inPars['omega'][iPar] * util.degToRad],
                period=[inPars['period'][iPar]],
                mjd_0=[pars.MJD_0],
                temp_star=[pars.TSTAR],
                rad_star=[pars.RSTAR],
                mass_star=[16],
                mass_compact=[1.4],
                f_m=[0.01],
                x1=[0.362]
            )
        elif inPars['orb'][iPar] == 'mo':
            systems = orbit.generate_systems(
                eccentricity=[inPars['eccentricity'][iPar]],
                phase_per=[0.663],
                inclination=[inPars['inclination'][iPar] * util.degToRad],  # 47-80
                omega=[inPars['omega'][iPar] * util.degToRad],
                period=[inPars['period'][iPar]],
                mjd_0=[pars.MJD_0],
                temp_star=[pars.TSTAR],
                rad_star=[pars.RSTAR],
                mass_star=[16],
                mass_compact=[1.4],
                f_m=[0.0024],
                x1=[0.120]
            )

        orbits = orbit.SetOfOrbits(phase_step=0.0005, color='r', systems=systems, mjd_pts=mjd_pts)

        pts_orbit = orbits.get_pts()
        dist_orbit = pts_orbit['distance']
        theta_orbit = pts_orbit['theta_ic']
        pos_orbit = pts_orbit['pos_3D']

        #######
        # Fit
        do_fit(
            label=inPars['label'][iPar],
            periods=periods,
            lgEdot_min=lgEdot_min,
            lgEdot_max=lgEdot_max,
            lgEdot_bins=inPars['lgEdot_bins'][iPar],
            lgSigma_min=lgSigma_min,
            lgSigma_max=lgSigma_max,
            lgSigma_bins=inPars['lgSigma_bins'][iPar],
            Dist=np.array(dist_orbit),
            ThetaIC=np.array(theta_orbit),
            Pos3D=pos_orbit,
            Vw=1500,
            AlphaSigma=inPars['AlphaSigma'][iPar],
            Mdot=inPars['Mdot'][iPar],
            Alpha=alphas
        )
