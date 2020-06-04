#!/usr/bin/python

import logging

from tgblib.absorption import Absorption
from tgblib.parameters import TSTAR, RSTAR

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == '__main__':

    # Table
    Abs = Absorption(Tstar=TSTAR, Rstar=RSTAR, read_table=False)
    Abs.ProduceLambdaTable(n_en=200, n_alpha=120, name='absorption_table.dat')
