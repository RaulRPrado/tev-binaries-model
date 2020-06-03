#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy import misc

from iminuit import Minuit

from tgblib import data


class SpectrumFit:
    def __init__(self, energy=None, spec=None, specErr=None):
        self.set_data(energy=energy, spec=spec, specErr=specErr)

    def set_data(self, energy=None, spec=None, specErr=None):
        if energy is not None:
            self.energy = copy.copy(energy)
        if spec is not None:
            self.spec = copy.copy(spec)
        if specErr is not None:
            self.specErr = copy.copy(specErr)

    def pl_flux(self, e, F, g, e0, e1, e2):
        n = F * math.pow(e0, -g) * (2 - g) / (math.pow(e2, 2 - g) - math.pow(e1, 2 - g))
        return n * math.pow(e, 2) * math.pow(e / e0, -g)

    def fit_power_law(self, Emin=None, Emax=None):
        self.Emin = Emin if Emin is not None else self.energy[0]
        self.Emax = Emax if Emax is not None else self.energy[-1]
        self.E0 = (self.Emin + self.Emax) / 2

        def least_square(f, g):
            model = [
                self.pl_flux(e=e, F=f, g=g, e0=self.E0, e1=self.Emin, e2=self.Emax) for e in self.energy
            ]
            chisq = [((f - m) / e)**2 for (f, e, m) in zip(self.spec, self.specErr, model)]
            return sum(chisq)

        minuit = Minuit(
            least_square,
            f=1e-12,
            fix_f=False,
            g=1.5,
            fix_g=False,
            limit_f=(1e-15, 1e-9),
            limit_g=(0, 3)
        )

        fmin, param = minuit.migrad()

        self.ChiSq = minuit.fval
        self.ndf = len(self.energy) - 2

        self.flux, self.fluxErr = param[0]['value'], param[0]['error']
        self.gamma, self.gammaErr = param[1]['value'], param[1]['error']
        cov = minuit.matrix()
        self.sigmaFlux = cov[0][0]
        self.sigmaGamma = cov[1][1]
        self.sigmaFG = cov[0][1]

        self.enFit = np.linspace(self.Emin, self.Emax, 100)
        self.specFit = [
            self.pl_flux(
                e=e,
                F=self.flux,
                g=self.gamma,
                e0=self.E0,
                e1=self.Emin,
                e2=self.Emax
            ) for e in self.enFit
        ]

    def plot_data(self, **kwargs):
        plt.errorbar(self.energy, self.spec, yerr=self.specErr, **kwargs)

    def plot_fit(self, **kwargs):
        plt.plot(self.enFit, self.specFit, **kwargs)

    def plot_fit_unc(self, nocor=False, **kwargs):

        def derivFlux(e):
            def specFlux(F):
                return self.pl_flux(e=e, F=F, g=self.gamma, e0=self.E0, e1=self.Emin, e2=self.Emax)
            return misc.derivative(func=specFlux, x0=self.flux)

        dfdF = [derivFlux(e) for e in self.enFit]

        def derivGamma(e):
            def specGamma(g):
                return self.pl_flux(e=e, F=self.flux, g=g, e0=self.E0, e1=self.Emin, e2=self.Emax)
            return misc.derivative(func=specGamma, x0=self.gamma)

        dfdG = [derivGamma(e) for e in self.enFit]

        def sigma(e, f, g):
            s2 = (math.fabs(f)**2) * self.sigmaFlux
            s2 += (math.fabs(g)**2) * self.sigmaGamma
            if not nocor:
                s2 += 2 * f * g * self.sigmaFG
            return math.sqrt(s2)

        self.uncFit = [sigma(e=e, f=f, g=g) for (e, f, g) in zip(self.enFit, dfdF, dfdG)]

        self.specFitUpper = [s + u for (s, u) in zip(self.specFit, self.uncFit)]
        self.specFitLower = [s - u for (s, u) in zip(self.specFit, self.uncFit)]

        plt.fill_between(x=self.enFit, y1=self.specFitUpper, y2=self.specFitLower, **kwargs)

    def get_norm(self, e):
        return self.pl_flux(e=e, F=self.flux, g=self.gamma, e0=self.E0, e1=self.Emin, e2=self.Emax)


if __name__ == '__main__':

    util.set_my_fonts(mode='talk')
    show = True

    mjd = [58079, 58101]

    for iper in range(2):

        data_en, data_fl, data_fl_er = data.get_data(iper)

        sf = SpectrumFit(energy=[e for e in data_en if e < 1e6],
                         spec=[f for (f, e) in zip(data_fl, data_en) if e < 1e6],
                         specErr=[f for (f, e) in zip(data_fl_er, data_en) if e < 1e6])

        sf.fit_power_law(Emin=3, Emax=30)


        ##########
        # NuSTAR
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel(r'$E^2\;\mathrm{d}N/\mathrm{d}E\;[\mathrm{erg\;s^{-1}\;cm^{-2}}]$')
        ax.set_xlabel(r'$E [\mathrm{keV}]$')

        # sf.plot_data(color='k', marker='o', linestyle='None')

        sf.plot_fit(color='k', linestyle='--', linewidth=2)

        sf.plot_fit_unc(color='k', linestyle='None', alpha=0.3)
        sf.plot_fit_unc(nocor=True, color='r', linestyle='None', alpha=0.3)

        plt.show()

        print('end')
        exit()

        x_fit = np.array([e * u.keV.to(u.keV) for e in data_en if e < 1e6])
        y_fit = np.array([f for (f, e) in zip(data_fl, data_en) if e < 1e6])
        yerr_fit = np.array([f for (f, e) in zip(data_fl_er, data_en) if e < 1e6])

        def PLFlux(e, F, g, e0, e1, e2):
            n = F * math.pow(e0, -g) * (2 - g) / (math.pow(e2, 2 - g) - math.pow(e1, 2 - g))
            return n * math.pow(e, 2) * math.pow(e / e0, -g)

        def least_square_nustar(f, g):
            model = [PLFlux(e=e, F=f, g=g, e0=10, e1=3, e2=30) for e in x_fit]
            chisq = [((f - m) / e)**2 for (f, e, m) in zip(y_fit, yerr_fit, model)]
            return sum(chisq)

        minuit = Minuit(least_square_nustar,
                        f=1e-12, fix_f=False,
                        g=1.5, fix_g=False,
                        limit_f=(1e-15, 1e-9),
                        limit_g=(0, 3))

        fmin, param = minuit.migrad()

        # print(fmin)
        # print(param)

        chisq_min = minuit.fval
        ndf = len(x_fit) - 2
        print(chisq_min / ndf)

        flux, flux_er = param[0]['value'], param[0]['error']
        gamma, gamma_er = param[1]['value'], param[1]['error']

        print('F=', flux, flux_er)
        print('g=', gamma, gamma_er)

        # en_fit = np.linspace(x_fit[0]*0.9, x_fit[-1]/0.9, 100)
        en_fit = np.linspace(3, 30, 100)
        flux_fit = [PLFlux(e, flux, gamma, 10, 3, 30) for e in en_fit]

        plt.plot(en_fit, flux_fit, c='k', ls='--')

        ylim = ax.get_ylim()
        ax.set_ylim(1e-13, ylim[1])
        ax.set_xlim(2e0, 4e1)

        if show:
            plt.show()
        else:
            plt.savefig('figures/DataNuStar_' + str(iper) + '.png', format='png',
                        bbox_inches='tight')
            plt.savefig('figures/DataNuStar_' + str(iper) + '.pdf', format='pdf',
                        bbox_inches='tight')
