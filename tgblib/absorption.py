#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import itertools
import time
from numpy import arccos
from numpy.linalg import norm
from itertools import combinations

import astropy.constants as const
import astropy.units as u

from tgblib import util
from tgblib import orbit
from tgblib import pulsar as psr


def SigmaGG(en0, en1, alpha):
    # en0, en1 in TeV
    # alpha in deg
    # return sigma in cm**2
    beta2 = 1
    beta2 -= 2 * (const.m_e * const.c**2)**2 / ((en0 * u.TeV) * (en1 * u.TeV) * (1 + math.cos(alpha * u.deg.to(u.rad))))
    if beta2 < 0:
        return 0
    beta = math.sqrt(beta2.to(1).value)
    sig = (3 * const.sigma_T / 16) * (1 - beta**2)
    sig *= (3 - beta**4)*math.log((1 + beta)/(1 - beta)) - 2 * beta * (2 - beta**2)
    return sig.to(u.cm**2).value


class Absorption(object):
    def __init__(self, Tstar, Rstar, read_table=True, name_table='absorption_table.txt',
                 verbose=False):
        self.Tstar = Tstar  # in K
        self.Rstar = Rstar  # in Rsun
        self.TableData = dict()
        self.verbose = verbose
        if read_table:
            self.ReadTable(name=name_table)

    def ProduceLambdaTable(self, name='absorption_table.txt', n_en=100, n_alpha=200):
        print('Producing table:', name)
        print('# of steps:', n_alpha * n_en)
        lg_en = math.log10((self.Tstar * u.K * const.k_B).to(u.eV).value)
        en_planck = np.logspace(lg_en - 2, lg_en + 2, 500)  # in eV

        table = open(name, 'w')

        # cos_list = np.linspace(-1, 1, n_alpha)
        # alpha_list = [u.rad.to(u.deg) * math.acos(c) for c in cos_list]  # in deg
        alpha_list = np.linspace(0, 179, n_alpha)
        en_list = np.logspace(-4, 1, n_en)  # in TeV

        start_time = time.time()
        step = 0
        for (alpha, en0) in itertools.product(alpha_list, en_list):
            if step % 100 == 0:
                print('Step ', step)
                if step > 0:
                    print('time/step [s] =', (time.time() - start_time) / step)
            sig_spec = [SigmaGG(en0=en0, en1=e * u.eV.to(u.TeV), alpha=alpha) * PlanckSpec(en=e, Tstar=self.Tstar)
                        for e in en_planck]
            sig_spec_int = np.trapz(x=en_planck, y=sig_spec)
            table.write(str(en0) + ' ' + str(alpha) + ' ' + str(sig_spec_int) + '\n')
            step += 1

        table.close()
        print('Done')

    def ReadTable(self, name):
        data = np.loadtxt(name, unpack=True)
        self.TableData = dict()

        for i in range(len(data[0])):
            en = data[0][i]
            alpha = data[1][i]
            spec = data[2][i]
            if en not in self.TableData.keys():
                self.TableData[en] = list()
            self.TableData[en].append([alpha, spec])

    def LambdaGG(self, en0, alpha, U):
        # en0 in TeV
        # alpha in deg
        # U in erg /cm3
        # return lambda in cm

        # Reading spec from table
        # Interpolating energy
        en_list = copy.copy(list(self.TableData.keys()))
        xp_en, fp_en = list(), list()
        for i in range(2):
            idx_en = np.argmin([(e - en0)**2 for e in en_list])

            # Interpolating alpha
            xp_alpha, fp_alpha = list(), list()
            alpha_list = copy.copy(self.TableData[en_list[idx_en]])
            for i in range(2):
                idx_alpha = np.argmin([(a[0] - alpha)**2 for a in alpha_list])
                xp_alpha.append(alpha_list[idx_alpha][0])
                fp_alpha.append(alpha_list[idx_alpha][1])
                alpha_list.pop(idx_alpha)

            xp_alpha, fp_alpha = zip(*sorted(zip(xp_alpha, fp_alpha)))
            IntSpec = np.interp(alpha, xp=xp_alpha, fp=fp_alpha)

            # xp_en.append(en_list[idx_en])
            # fp_en.append(IntSpec)
            xp_en.append(math.log10(en_list[idx_en]))
            fp_en.append(IntSpec)
            en_list.pop(idx_en)

        xp_en, fp_en = zip(*sorted(zip(xp_en, fp_en)))
        IntSpec = np.interp(math.log10(en0), xp=xp_en, fp=fp_en)

        Ufac = U / PlanckEnergyDensity(Tstar=self.Tstar)  # unitless
        if IntSpec <= 0:
            return 1e80
        return 1 / (IntSpec * Ufac * (1 + math.cos(alpha * u.deg.to(u.rad))))

    def TauGG(self, en, obs, pos, dx_ratio=0.01, min_tau=3e-4):
        # en in TeV - gamma-ray energy
        # T in K
        # obs, pos - 3D vectors normalized in AU
        if en < 1e-6:
            return 0
        if self.verbose:
            print('E/TeV = ', en)
        start = time.time()
        if self.LambdaGG(en0=en, alpha=1, U=1) > 1e79:
            return 0
        tau, dtau = 0, 100
        x = pos
        while (norm(x) <= 20 * norm(pos)) or (dtau > min_tau):
            dx = (obs / norm(obs)) * dx_ratio * norm(x)
            U = psr.PhotonDensity(Tstar=self.Tstar, Rstar=self.Rstar, d=norm(x))
            alpha = arccos(np.dot(x, obs) / norm(x)) * u.rad.to(u.deg)
            lamb = self.LambdaGG(en0=en, alpha=alpha, U=U)
            dtau = norm(dx) * u.au.to(u.cm) / lamb if lamb < 1e79 else 0
            tau += dtau
            x = x + dx
        if self.verbose:
            print('dt = ', time.time() - start)
        return tau


def LambdaGG(en0, Tstar, alpha, U, no_int=False):
    # en0 in TeV
    # T in K
    # alpha in deg
    # U in erg /cm3
    # return lambda in cm
    if not no_int:
        lg_en = math.log10((Tstar * u.K * const.k_B).to(u.eV).value)
        en = np.logspace(lg_en - 2, lg_en + 2, 500)
        sig_spec = [SigmaGG(en0=en0, en1=e * u.eV.to(u.TeV), alpha=alpha) * PlanckSpec(en=e, Tstar=Tstar)
                    for e in en]
        Ufac = U / PlanckEnergyDensity(Tstar=Tstar)  # unitless
        sig_spec_int = np.trapz(x=en, y=sig_spec)
        if sig_spec_int <= 0:
            return 1e80
        return 1 / (sig_spec_int * Ufac * (1 + math.cos(alpha * u.deg.to(u.rad))))
    else:
        en1 = (Tstar * u.K * const.k_B).to(u.TeV).value
        sig = SigmaGG(en0=en0, en1=en1, alpha=alpha)
        if sig == 0:
            return 1e80
        n = U / (en1 * u.TeV).to(u.erg).value
        return 1 / (n * sig * (1 + math.cos(alpha * u.deg.to(u.rad))))


def TauGG(en, Tstar, Rstar, obs, pos, dx_ratio=0.001, min_tau=1e-6, no_int=False):
    # en in TeV - gamma-ray energy
    # T in K
    # obs, pos - 3D vectors normalized in AU
    print(no_int, en)
    start = time.time()
    if LambdaGG(en0=en, Tstar=Tstar, alpha=1, U=1) > 1e79:
        return 0
    tau, dtau = 0, 100
    x = pos
    while (norm(x) <= 20 * norm(pos)) or (dtau > min_tau):
        dx = (obs / norm(obs)) * dx_ratio * norm(x)
        U = psr.PhotonDensity(Tstar=Tstar, Rstar=Rstar, d=norm(x))
        alpha = arccos(np.dot(x, obs) / norm(x)) * u.rad.to(u.deg)
        lamb = LambdaGG(en0=en, Tstar=Tstar, alpha=alpha, U=U, no_int=no_int)
        dtau = norm(dx) * u.au.to(u.cm) / lamb if lamb < 1e79 else 0
        tau += dtau
        x = x + dx
    print('dt = ', time.time() - start)
    return tau


def PlanckEnergyDensity(Tstar):
    # T in K
    # return in erg /cm3
    return (2 * const.sigma_sb * (Tstar * u.K)**4 / const.c).to(u.erg / u.cm**3).value


def PlanckSpec(en, Tstar):
    # en in eV
    # T in K
    n = 4 * math.pi / (const.h * const.c)**3
    n *= (en * u.eV)**2 / (math.exp((en * u.eV / (const.k_B * Tstar * u.K)).to(1).value) - 1)
    return n.to(1 / u.cm**3 / u.eV).value


if __name__ == '__main__':

    # Table
    Abs = Absorption(Tstar=3e4, Rstar=7.8, read_table=False)
    Abs.ProduceLambdaTable(n_en=200, n_alpha=120, name='absorption_table.txt')
    # # obs = np.array([0, 0, -1])
    # print(Abs.TauGG(en=1, obs=obs, pos=np.array([0, 2, 1.5]), dx_ratio=0.05, min_tau=1e-3))
    # print(TauGG(en=1, Tstar=3e4, obs=obs, pos=np.array([0, 2, 1.5]), no_int=True, dx_ratio=0.05, min_tau=1e-3))
    # print(TauGG(en=1, Tstar=3e4, obs=obs, pos=np.array([0, 2, 1.5]), no_int=False, dx_ratio=0.05, min_tau=1e-3))

    # # Ploting energy and alpha dependence
    # Abs = Absorption(Tstar=3e4, Rstar=7.8, read_table=True)

    # # Alpha
    # alpha = np.linspace(0, 175, 40)

    # plt.figure(figsize=(8, 6), tight_layout=True)
    # ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_ylabel(r'$\lambda_{\gamma \gamma}$')
    # ax.set_xlabel(r'$\alpha$ [deg]')

    # ax.plot(alpha, [Abs.LambdaGG(en0=0.1, U=1, alpha=a) for a in alpha],
    #         label='E=0.2 TeV', color='k', linestyle='-')
    # ax.plot(alpha, [Abs.LambdaGG(en0=1.2, U=1, alpha=a) for a in alpha],
    #         label='E=1.2 TeV', color='k', linestyle='--')

    # ax.plot(alpha, [LambdaGG(en0=0.1, U=1, Tstar=3e4, alpha=a) for a in alpha],
    #         label='E=0.2 TeV', color='r', linestyle='-')
    # ax.plot(alpha, [LambdaGG(en0=1.2, U=1, Tstar=3e4, alpha=a) for a in alpha],
    #         label='E=1.2 TeV', color='r', linestyle='--')

    # ax.legend(frameon=False)

    # plt.show()

    # # Energy
    # energy = np.logspace(-1, 0.9, 40)

    # plt.figure(figsize=(8, 6), tight_layout=True)
    # ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylabel(r'$\lambda_{\gamma \gamma}$')
    # ax.set_xlabel(r'$E$ [TeV]')

    # ax.plot(energy, [Abs.LambdaGG(en0=e, U=1, alpha=120) for e in energy],
    #         label=r'$\alpha=120$ deg', color='k', linestyle='-')
    # ax.plot(energy, [Abs.LambdaGG(en0=e, U=1, alpha=60) for e in energy],
    #         label=r'$\alpha=60$ deg', color='k', linestyle='--')

    # ax.plot(energy, [LambdaGG(en0=e, U=1, Tstar=3e4, alpha=120) for e in energy],
    #         label=r'$\alpha=120$ deg', color='r', linestyle='-')
    # ax.plot(energy, [LambdaGG(en0=e, U=1, Tstar=3e4, alpha=60) for e in energy],
    #         label=r'$\alpha=60$ deg', color='r', linestyle='--')

    # ax.legend(frameon=False)

    # plt.show()
