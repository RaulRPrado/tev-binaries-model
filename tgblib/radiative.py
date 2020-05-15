#!/usr/bin/python

import numpy as np
import math

import astropy.units as u
import astropy.constants as const


def TauAcc(E, B, eff=1):
    # Eq 22 of Dubus 2013
    # E in TeV
    # B in G
    return 0.1 * eff * E / B


def TauIC(E, U, T):
    # E in TeV
    # U in erg / cm3
    # T in K
    tt = const.k_B * T * u.K / (const.m_e * const.c**2)
    gamma = E * u.TeV / (const.m_e * const.c**2)
    g_dot = 5.5e17 * (tt**3) * gamma
    g_dot *= math.log(1 + 0.55 * gamma * tt) / (1 + 25 * tt * gamma)
    g_dot *= 1 + 1.4 * gamma * tt / (1 + 12 * (gamma**2)*(tt**2))
    Ufac = (U * u.erg / u.cm**3) / (4 * const.sigma_sb * (T * u.K)**4 / const.c)
    return gamma.to(1) / g_dot.to(1) / Ufac


def TauSyn(E, B):
    # E in TeV
    # B in G
    # From Eq 23 of Dubus 2013
    return 400 * (1. / E) * (1 / B)**2


def Ebreak(B, T, U):
    # B in G
    # T in K
    # U in erg /cm**3
    ####
    # 1st step
    en = np.logspace(-7, 3, 100)
    tdif = [abs(TauSyn(E=e, B=B) - TauIC(E=e, T=T, U=U)) for e in en]
    idx = np.argmin(tdif)
    ebr = en[idx]
    # 2nd step
    lge = math.log10(ebr)
    en = np.logspace(lge - 0.05, lge + 0.05, 100)
    tdif = [abs(TauSyn(E=e, B=B) - TauIC(E=e, T=T, U=U)) for e in en]
    idx = np.argmin(tdif)
    ebr = en[idx]
    # 3rd step
    lge = math.log10(ebr)
    en = np.logspace(lge - 0.0005, lge + 0.0005, 100)
    tdif = [abs(TauSyn(E=e, B=B) - TauIC(E=e, T=T, U=U)) for e in en]
    idx = np.argmin(tdif)
    ebr = en[idx]
    return en[idx] if tdif[idx] < 10 else 0


def Emax(B, eff=1):
    # Eq. 24 from Dubus 2013
    return 60 * math.pow(eff * B, -0.5)


def Esyn(E, B):
    # E in TeV
    # B in G
    return 5 * 1e-8 * B * (E**2)


def bracket_sig(x):
    r = 1 - 2 * (x + 1) / x**2
    r *= math.log(2 * x + 1)
    r += 0.5 + 4 / x - 1 / (2 * ((2*x + 1)**2))
    return r


def sigma_kn(E, T):
    eph = 2.7 * const.k_B * T * u.K
    gamma = E * u.TeV / (const.m_e * const.c**2)
    x = eph * gamma / (const.m_e * const.c**2)
    return (3./8) * const.sigma_T * bracket_sig(x.to(1)) / x.to(1)
    # return (3./8) * const.sigma_T * (math.log(2*x.to(1))+0.5) / x.to(1)


def tau_ic_old(E, U):
    t = 3e7 * u.yr.to(u.s)
    t /= U * u.erg.to(u.eV)
    t /= 100 * E
    return t


def tau_kn_new(E, U, T):
    gamma = E * u.TeV / (const.m_e * const.c**2)
    t = tau_ic(E=E, U=U) * const.sigma_T / sigma_kn(E=E, T=T)
    #
    # eph = 2.7 * const.k_B * T * u.K
    # t = 1e6 * u.yr.to(u.s) * const.sigma_T / sigma_kn(E=E, T=T)
    # t /= U / eph.to(u.erg).value
    #
    # eph = 2.7 * const.k_B * T * u.K
    # t = 1e-20 * 3e7 * u.yr.to(u.s) / U / eph.to(u.eV).value
    # t *= const.sigma_T / sigma_kn(E=E, T=T)
    return t


def tau_kn(E, T, U):
    # From BG70
    # E in TeV
    # T in K
    # U in erg / c**3
    # return t in s
    de = (const.sigma_T / 16) * (const.m_e * const.c * const.k_B * T * u.K)**2 / const.hbar**3
    a = 4 * E * u.TeV * const.k_B * T * u.K / (const.m_e**2 * const.c**4)
    de *= (math.log(a.to(1)) - 5/6. - 1.147)
    # The calculation by BG70 includes the energy density
    # In order to multiply by the energy density, we need
    # to correct the unit
    # density_unit is the unit of density from BG70 calculations
    # that we need to convert to erg / cm3
    # (This was cross-checked with Eq 28 of Dubus 2013)
    density_unit = const.c * const.k_B**2 * u.K**2 * const.m_e**2 / (const.hbar**3)
    fac = density_unit.to(u.erg / u.cm**3).value
    de *= U / fac
    t = E * u.TeV / de
    return t.to(u.s).value
