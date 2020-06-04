#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import logging

import tgblib.absorption as abso
from tgblib.parameters import TSTAR, RSTAR

logging.getLogger().setLevel(logging.DEBUG)


def test_make_table():
    Abs = abso.Absorption(Tstar=TSTAR, Rstar=RSTAR, read_table=False)
    Abs.ProduceLambdaTable(n_en=10, n_alpha=10, name='absorption_table_test.dat')


def test_tau():
    obs = np.array([0, 0, -1])
    Abs = abso.Absorption(
        Tstar=TSTAR,
        Rstar=RSTAR,
        read_table=True,
        name_table='absorption_table_test.dat'
    )

    logging.info('Tau from class')
    tau0 = Abs.TauGG(en=1, obs=obs, pos=np.array([0, 2, 1.5]), dx_ratio=0.05, min_tau=1e-3)
    logging.info('Tau0 = {}'.format(tau0))

    logging.info('Tau from external function - no_int=True')

    tau1 = abso.TauGG(
        en=1,
        Tstar=TSTAR,
        Rstar=RSTAR,
        obs=obs,
        pos=np.array([0, 2, 1.5]),
        no_int=True,
        dx_ratio=0.05,
        min_tau=1e-3
    )
    logging.info('Tau1 = {}'.format(tau1))

    logging.info('Tau from external function - no_int=False')
    tau2 = abso.TauGG(
        en=1,
        Tstar=TSTAR,
        Rstar=RSTA,
        obs=obs,
        pos=np.array([0, 2, 1.5]),
        no_int=False,
        dx_ratio=0.05,
        min_tau=1e-3
    )
    logging.info('Tau2 = {}'.format(tau2))


def test_alpha():
    # Ploting energy and alpha dependence
    Abs = abso.Absorption(
        Tstar=TSTAR,
        Rstar=RSTAR,
        read_table=True,
        name_table='absorption_table_test.dat'
    )

    # Alpha
    alpha = np.linspace(0, 175, 40)

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylabel(r'$\lambda_{\gamma \gamma}$')
    ax.set_xlabel(r'$\alpha$ [deg]')

    ax.plot(
        alpha,
        [Abs.LambdaGG(en0=0.1, U=1, alpha=a) for a in alpha],
        label='E=0.2 TeV',
        color='k',
        linestyle='-'
    )
    ax.plot(
        alpha,
        [Abs.LambdaGG(en0=1.2, U=1, alpha=a) for a in alpha],
        label='E=1.2 TeV',
        color='k',
        linestyle='--'
    )

    ax.plot(
        alpha,
        [abso.LambdaGG(en0=0.1, U=1, Tstar=TSTAR, alpha=a) for a in alpha],
        label='E=0.2 TeV',
        color='r',
        linestyle='-'
    )
    ax.plot(
        alpha,
        [abso.LambdaGG(en0=1.2, U=1, Tstar=TSTAR, alpha=a) for a in alpha],
        label='E=1.2 TeV',
        color='r',
        linestyle='--'
    )

    ax.legend(frameon=False)

    plt.show()


def test_energy():
    # Energy
    Abs = abso.Absorption(
        Tstar=TSTAR,
        Rstar=RSTAR,
        read_table=True,
        name_table='absorption_table_test.dat'
    )

    energy = np.logspace(-1, 0.9, 40)

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\lambda_{\gamma \gamma}$')
    ax.set_xlabel(r'$E$ [TeV]')

    ax.plot(
        energy,
        [Abs.LambdaGG(en0=e, U=1, alpha=120) for e in energy],
        label=r'$\alpha=120$ deg',
        color='k',
        linestyle='-'
    )
    ax.plot(
        energy,
        [Abs.LambdaGG(en0=e, U=1, alpha=60) for e in energy],
        label=r'$\alpha=60$ deg',
        color='k',
        linestyle='--'
    )

    ax.plot(
        energy,
        [abso.LambdaGG(en0=e, U=1, Tstar=TSTAR, alpha=120) for e in energy],
        label=r'$\alpha=120$ deg',
        color='r',
        linestyle='-'
    )
    ax.plot(
        energy,
        [abso.LambdaGG(en0=e, U=1, Tstar=TSTAR, alpha=60) for e in energy],
        label=r'$\alpha=60$ deg',
        color='r',
        linestyle='--'
    )

    ax.legend(frameon=False)

    plt.show()


if __name__ == '__main__':

    # test_make_table()
    # test_tau()
    # test_alpha()
    test_energy()
    pass
