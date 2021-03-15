#!/usr/bin/python

import logging

from tgblib.absorption import Absorption
from tgblib.parameters import TSTAR_LS, RSTAR_LS

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == '__main__':

    # Table
    Abs = Absorption(Tstar=TSTAR_LS, Rstar=RSTAR_LS, read_table=False)
    Abs.ProduceLambdaTable(n_en=200, n_alpha=120, name='absorption_table_ls5039.dat')
