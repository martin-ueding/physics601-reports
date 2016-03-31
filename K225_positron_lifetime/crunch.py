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
import time_gauge

TEMP_PATTERN = re.compile('in-(\d+(?:,\d+)?)-(\d+(?:,\d+)?)C\.txt')


def dandify_plot():
    '''
    Common operations to make matplotlib plots look nicer.
    '''
    pl.grid(True)
    pl.margins(0.05)
    pl.tight_layout()


def get_temp(filename):
    '''
    Retrieves the temperatures stored in the filename itself.

    :param str filename: Filename or full path
    :returns tuple(str): Tuple with upper and lower temperature.

    >>> get_temp('in-102,5-104,2C.txt')
    (102.5, 104.2)
    >>> get_temp('in-102-104,2C.txt')
    (102.0, 104.2)
    >>> get_temp('in-102,5-104C.txt')
    (102.5, 104.0)
    '''
    basename = os.path.basename(filename)
    m = TEMP_PATTERN.match(basename)
    if m:
        first = float(m.group(1).replace(',', '.'))
        second = float(m.group(2).replace(',', '.'))

        return (first, second)

    return None


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
    dandify_plot()
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


def redraw_count(a):
    '''
    Takes a ``np.array`` with counts and re-draws the counts from the implicit
    Gaussian distribution with width ``sqrt(N)``.
    '''
    out = [random.gauss(x, np.sqrt(x)) for x in a]
    return np.array(out).reshape(a.shape)


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

    spectrum.job_lifetime_spectra(T)
    prepare_files(T)
    time_gauge.job_time_gauge(T)


    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
