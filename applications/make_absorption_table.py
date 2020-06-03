#!/usr/bin/python

from tgblib.absorption import Absorption


if __name__ == '__main__':

    # Table
    Abs = Absorption(Tstar=3e4, Rstar=7.8, read_table=False)
    Abs.ProduceLambdaTable(n_en=200, n_alpha=120, name='absorption_table.txt')
