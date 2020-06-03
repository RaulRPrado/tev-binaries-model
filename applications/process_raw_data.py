#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import math
import logging

import astropy.units as u
from astropy.io import ascii


if __name__ == '__main__':
    '''
    Energies in keV
    Integrated fluxes given in erg cm-2 s-1
    Warning: NuSTAR files are mislabed - flux_error_hi is the model fit.
    Use only flux_error_lo as flux_err.
    '''
    logging.warning(
        'Warning: NuSTAR files are mislabed - flux_error_hi is the model fit. '
        'Use only flux_error_lo as flux_err.'
    )

    # Period 0 - Nov 2017
    logging.info('Period 0 - Nov. 2017')

    fileNameNuSTAR = 'data/nov_joint_pow_3-30_keV_3sig_new.csv'
    dataNuSTAR = ascii.read(fileNameNuSTAR, format='basic')

    fileNameVTS = 'data/VTS_58073-58083.ecsv'
    dataVTS = ascii.read(fileNameVTS, format='basic')

    outData = dict()
    outData['energy'] = list()
    outData['flux'] = list()
    outData['flux_err'] = list()

    convFluxNuSTAR = u.keV.to(u.erg)
    for d in dataNuSTAR:
        outData['energy'].append(d['energy'])
        outData['flux'].append(d['flux'] * convFluxNuSTAR)
        outData['flux_err'].append(d['flux_error_lo'] * convFluxNuSTAR)

    convEnergyVTS = u.TeV.to(u.keV)
    convFluxVTS = u.TeV.to(u.erg)
    for d in dataVTS:
        if d['dnde_error'] == 0.:
            continue
        en = 10**d['lge']  # TeV
        outData['energy'].append(en * convEnergyVTS)
        outData['flux'].append(d['dnde'] * (en**2) * convFluxVTS)
        dnde_err = (d['dnde_error_hi'] + d['dnde_error_lo']) / 2
        outData['flux_err'].append(dnde_err * (en**2) * convFluxVTS)

    ascii.write(outData, 'data/HESS_J0632_0.csv', format='basic', overwrite=True)

    # Period 1 - Dec 2017
    logging.info('Period 1 - Dec. 2017')

    fileNameNuSTAR = 'data/dec_joint_pow_3-30_keV_3sig_new.csv'
    dataNuSTAR = ascii.read(fileNameNuSTAR, format='basic')

    fileNameVTS = 'data/VTS_58101-58103.ecsv'
    dataVTS = ascii.read(fileNameVTS, format='basic')

    outData = dict()
    outData['energy'] = list()
    outData['flux'] = list()
    outData['flux_err'] = list()

    convFluxNuSTAR = u.keV.to(u.erg)
    for d in dataNuSTAR:
        outData['energy'].append(d['energy'])
        outData['flux'].append(d['flux'] * convFluxNuSTAR)
        outData['flux_err'].append(d['flux_error_lo'] * convFluxNuSTAR)

    convEnergyVTS = u.TeV.to(u.keV)
    convFluxVTS = u.TeV.to(u.erg)
    for d in dataVTS:
        if d['dnde_error'] == 0.:
            continue
        en = 10**d['lge']  # TeV
        outData['energy'].append(en * convEnergyVTS)
        outData['flux'].append(d['dnde'] * (en**2) * convFluxVTS)
        dnde_err = (d['dnde_error_hi'] + d['dnde_error_lo']) / 2
        outData['flux_err'].append(dnde_err * (en**2) * convFluxVTS)

    ascii.write(outData, 'data/HESS_J0632_1.csv', format='basic', overwrite=True)

    # Period 2 - Dec 2019
    logging.info('Period 2 - Dec. 2019')

    fileNameNuSTAR = 'data/nu30502017002_E3_20_SED_FPMA.csv'
    dataNuSTAR = ascii.read(fileNameNuSTAR, format='basic')

    fileNameVTS = 'data/1912_BDTmoderate2tel_SED.csv'
    dataVTS = ascii.read(fileNameVTS, format='basic')

    outData = dict()
    outData['energy'] = list()
    outData['flux'] = list()
    outData['flux_err'] = list()

    convFluxNuSTAR = u.keV.to(u.erg)
    for d in dataNuSTAR:
        outData['energy'].append(d['energy'])
        outData['flux'].append(d['flux'] * convFluxNuSTAR)
        outData['flux_err'].append(d['flux_error'] * convFluxNuSTAR)

    convEnergyVTS = u.TeV.to(u.keV)
    convFluxVTS = u.TeV.to(u.erg)
    for d in dataVTS:
        if d['dnde_error'] == 0.:
            continue
        outData['energy'].append(d['energy'] * convEnergyVTS)
        outData['flux'].append(d['dnde'] * (d['energy']**2) * convFluxVTS)
        outData['flux_err'].append(d['dnde_error'] * (d['energy']**2) * convFluxVTS)

    ascii.write(outData, 'data/HESS_J0632_2.csv', format='basic', overwrite=True)

    # Period 3 - Jan 2020
    logging.info('Period 3 - Jan. 2020')

    fileNameVTS = 'data/2001_BDTmoderate2tel_SED.csv'
    dataVTS = ascii.read(fileNameVTS, format='basic')

    outData = dict()
    outData['energy'] = list()
    outData['flux'] = list()
    outData['flux_err'] = list()

    convEnergyVTS = u.TeV.to(u.keV)
    convFluxVTS = u.TeV.to(u.erg)
    for d in dataVTS:
        if d['dnde_error'] == 0.:
            continue
        outData['energy'].append(d['energy'] * convEnergyVTS)
        outData['flux'].append(d['dnde'] * (d['energy']**2) * convFluxVTS)
        outData['flux_err'].append(d['dnde_error'] * (d['energy']**2) * convFluxVTS)

    ascii.write(outData, 'data/HESS_J0632_3.csv', format='basic', overwrite=True)

    # Period 4 - Feb 2020
    logging.info('Period 4 - Feb. 2020')

    fileNameNuSTAR = 'data/nu30502017004_E3_20_SED_FPMA.csv'
    dataNuSTAR = ascii.read(fileNameNuSTAR, format='basic')

    fileNameVTS = 'data/2002_BDTmoderate2tel_SED.csv'
    dataVTS = ascii.read(fileNameVTS, format='basic')

    outData = dict()
    outData['energy'] = list()
    outData['flux'] = list()
    outData['flux_err'] = list()

    convFluxNuSTAR = u.keV.to(u.erg)
    for d in dataNuSTAR:
        outData['energy'].append(d['energy'])
        outData['flux'].append(d['flux'] * convFluxNuSTAR)
        outData['flux_err'].append(d['flux_error'] * convFluxNuSTAR)

    convEnergyVTS = u.TeV.to(u.keV)
    convFluxVTS = u.TeV.to(u.erg)
    for d in dataVTS:
        if d['dnde_error'] == 0.:
            continue
        outData['energy'].append(d['energy'] * convEnergyVTS)
        outData['flux'].append(d['dnde'] * (d['energy']**2) * convFluxVTS)
        outData['flux_err'].append(d['dnde_error'] * (d['energy']**2) * convFluxVTS)

    ascii.write(outData, 'data/HESS_J0632_4.csv', format='basic', overwrite=True)
