#!/usr/bin/python

import numpy as np
import math

import astropy.units as u
import astropy.constants as const


def comp_edot(P, Pdot):
    Edot = 4 * pow(math.pi, 2)
    Edot *= 1e45  # g * cm2
    Edot *= Pdot / pow(P, 3)
    Edot *= (u.g * u.cm * u.cm / u.s / u.s / u.s).to(u.erg / u.s)
    return Edot


def Rshock(Edot, Mdot, Vw, D):
    # Edot in erg /s
    # Mdot in Msun / yr
    # Vw in km / s
    # D in AU
    # Eq 3 of Takata 2017
    eta = Edot * u.erg / u.s
    eta /= Mdot * u.Msun / u.yr
    eta /= Vw * u.km / u.s
    eta /= const.c
    eta = eta.to(1)
    Rs = math.sqrt(eta) / (1 + math.sqrt(eta))
    return D * Rs


def Rshock_Dubus(Edot, Mdot, Vw, D):
    # Edot in erg /s
    # Mdot in Msun / yr
    # Vw in km / s
    # D in AU
    eta = 0.05 * (Edot / 1e36) * (1e-7 / Mdot) * (1000 / Vw)
    ratio = pow(eta, 0.5) / (1 + pow(eta, 0.5))
    return ratio * D


def comp_age(P, Pdot):
    return u.s.to(u.yr) * P / (2 * Pdot)


def comp_b_ns(P, Pdot):
    return 3.2e19 * pow(P * Pdot, 0.5)


def SigmaRs(sig_l, Rs, alpha=1):
    return sig_l * pow(1e6 * u.m.to(u.au) / Rs, alpha)


def B2_KC(Edot, Rs, sigma):
    B1 = B1_KC(Edot, Rs, sigma)
    # Eq 4.11 of KC 1984a
    s = sigma
    s2 = sigma**2
    sp = sigma + 1
    sp2 = sp**2
    u2 = (8 * s2 + 10 * s + 1) / (16 * sp)
    u2 += math.sqrt(64 * s2 * sp2 + 20 * s * sp + 1) / (16 * sp)
    B2 = B1 * math.sqrt(1 + 1 / u2)
    return B2


def B1_KC(Edot, Rs, sigma):
    # Eq 2.2 of KC 1984b
    # Edot in erg / s
    # Rs in AU
    # return B in G
    B1_sq = Edot * (u.erg / u.s) * sigma
    B1_sq /= (Rs * u.au)**2
    B1_sq /= const.c * (1 + sigma)
    # converting B to natural units
    B1_sq = math.sqrt(8 * math.pi) * B1_sq.to(u.erg / u.cm**3)
    return math.sqrt(B1_sq.value)


def B2_KC_SmallSigma(Edot, Rs, sigma):
    # Eq. 4.15d of KC 1984a
    return 3 * (1 - 4 * sigma) * B1_KC(Edot, Rs, sigma)


def B2_KC_LargeSigma(Edot, Rs, sigma):
    # Eq. 4.14c of KC 1984a
    return (1 + 1 / (2 * sigma)) * B1_KC(Edot, Rs, sigma)


def B2_Dubus(Edot, Rs, sigma):
    B = 1.7 * pow(Edot / 1e37, 0.5)
    B *= pow(sigma / 1e-3, 0.5)
    B *= (1e12 * u.cm.to(u.au) / Rs)
    return B


def comp_b_edot_from_sig_l_old(Edot, Rs, sig_l, alpha=1):
    sigma = comp_sigma(sig_l, Rs, alpha)
    return comp_b_edot_old(Edot, Rs, sigma)


def comp_b_edot_from_sig_l(Edot, Rs, sig_l, alpha=1):
    sigma = comp_sigma(sig_l, Rs, alpha)
    return comp_b_edot(Edot, Rs, sigma)


def PhotonDensity(Tstar, Rstar, d):
    # Tstar in K
    # Rstar in Rsun
    # d in AU
    U = (const.sigma_sb / const.c)
    U *= pow(Tstar * u.K, 4)
    U *= pow(Rstar * u.Rsun, 2)
    U *= 1. / pow(d * u.au, 2)
    return U.to(u.erg / u.cm**3).value


def VwPolar(V0, Vinf, Rstar, r):
    # V's in km /s
    # Rstar in Rsun
    # r in AU
    return V0 + (Vinf - V0) * (1 - Rstar * u.Rsun.to(u.au) / r)


def VwDisk(V0, Rstar, r, m=0.5):
    # V0 in km / s
    # Rstar in Rsun
    # r in AU
    return V0 * math.pow(r / (Rstar * u.Rsun.to(u.au)), m)


def RshockPolarWind(Edot, Mdot, D, V0, Vinf, Rstar):
    # Edot in erg / s
    # Mdot in Msun / yr
    # D in AU
    # V's in km / s
    # Rstar in Rsun
    ratio_array = np.linspace(0.005, 0.995, 5000)
    dif = list()
    for r in ratio_array:
        Rs = r * D
        if D - Rs <= Rstar * u.Rsun.to(u.au):
            continue
        Vw = VwPolar(V0=V0, Vinf=Vinf, Rstar=Rstar, r=D - Rs)
        Rs2 = Rshock(Edot=Edot, Mdot=Mdot, Vw=Vw, D=D)
        dif.append((Rs-Rs2)**2)

    idx = np.argmin(dif)
    return ratio_array[idx] * D


def RshockDiskWind(Edot, Mdot, D, V0, Rstar, m):
    # Edot in erg / s
    # Mdot in Msun / yr
    # D in AU
    # V's in km / s
    # Rstar in Rsun
    ratio_array = np.linspace(0.005, 0.995, 5000)
    dif = list()
    for r in ratio_array:
        Rs = r * D
        if D - Rs <= Rstar * u.Rsun.to(u.au):
            continue
        Vw = VwDisk(V0=V0, Rstar=Rstar, r=D - Rs, m=m)
        Rs2 = Rshock(Edot=Edot, Mdot=Mdot, Vw=Vw, D=D)
        dif.append((Rs-Rs2)**2)

    idx = np.argmin(dif)
    return ratio_array[idx] * D
