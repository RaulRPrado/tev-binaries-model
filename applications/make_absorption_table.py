#!/usr/bin/python

from tgblib.absorption import Absorption
from tgblib.parameters import TSTAR, RSTAR


if __name__ == '__main__':

    # Table
    Abs = Absorption(Tstar=TSTAR, Rstar=RSTAR, read_table=False)
    Abs.ProduceLambdaTable(n_en=200, n_alpha=120, name='absorption_table.txt')
