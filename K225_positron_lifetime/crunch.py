#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import json
import os
import random
import re
import sys
import glob

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
import spectrum
import conf
import temperature
import time_gauge



def prepare_for_pgf(filename, lower=0, upper=8000, error=False):
    '''
    Converts raw data for use with pgfplots, reduces data.
    '''
    data = np.loadtxt('Data/{}.txt'.format(filename))
    channel = data[:,0]
    counts = data[:,1]
    step = 10
    channel_sel = channel[lower:upper:step]
    counts_sel = counts[lower:upper:step]

    if error:
        to_save = bootstrap.pgfplots_error_band(channel_sel,
                                                counts_sel,
                                                np.sqrt(counts_sel))
    else:
        to_save = np.column_stack([channel_sel, counts_sel])
    np.savetxt('_build/xy/{}.txt'.format(filename), to_save)

    pl.plot(channel, counts, linestyle="none", marker="o")
    conf.dandify_plot()
    pl.savefig('_build/mpl-channel-counts-{}.pdf'.format(filename))
    pl.clf()


def prepare_files(T):
    prepare_for_pgf('lyso-li', error=True)
    prepare_for_pgf('lyso-re', error=True)
    prepare_for_pgf('na-li', error=True)
    prepare_for_pgf('na-re', error=True)
    prepare_for_pgf('na-511-re')
    prepare_for_pgf('na-511-li')
    prepare_for_pgf('na-1275-li')


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
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    options = parser.parse_args()

    slope_val = time_gauge.job_time_gauge(T)
    indium_spectra = spectrum.indium_lifetime_spectra(T, slope_val)
    temperature.job_temperature_dependence(T, indium_spectra)

    prepare_files(T)


    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
