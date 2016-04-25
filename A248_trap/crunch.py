#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import itertools
import json
import os
import pprint
import random
import sys

import matplotlib.pyplot as pl
import numpy as np
import scipy.interpolate
import scipy.misc
import scipy.ndimage.filters
import scipy.optimize as op
import scipy.stats
import mpl_toolkits.mplot3d.axes3d as p3

from unitprint2 import siunitx
import bootstrap
import trek


def dandify_plot():
    pl.margins(0.05)
    pl.tight_layout()
    pl.grid(True)
    pl.legend(loc='best')


def job_some_osci(T):
    x1, y1, x2, y2 = trek.load_dir('0020')
    B_x1, B_y1, B_x2, B_y2 = trek.load_dir('0019')

    fig, ax1 = pl.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x1, y1, color='blue', label='Spectrum')
    ax1.plot(B_x1, B_y1, color='green', label='Spectrum')
    ax2.plot(x2, y2, color='red', label='MOT without B')
    ax2.plot(B_x2, B_y2, color='red', label='MOT with B')
    ax2.plot(B_x2, B_y2 - y2, color='orange', label='MOT signal')

    dandify_plot()
    pl.savefig('_build/mpl-20.pdf')
    pl.savefig('_build/mpl-20.png')
    pl.clf()


def test_keys(T):
    '''
    Testet das dict auf Schlüssel mit Bindestrichen.
    '''
    dash_keys = []
    for key in T:
        if '-' in key:
            dash_keys.append(key)

    if len(dash_keys) > 0:
        print()
        print('**************************************************************')
        print('* Es dürfen keine Bindestriche in den Schlüsseln für T sein! *')
        print('**************************************************************')
        print()
        print('Folgende Schlüssel enthalten Bindestriche:')
        for dash_key in dash_keys:
            print('-', dash_key)
        print()
        sys.exit(100)


def main():
    T = {}

    # We use bootstrap and obtain different results every single time. This is
    # bad, therefore we fix the seed here.
    random.seed(0)

    job_some_osci(T)

    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
