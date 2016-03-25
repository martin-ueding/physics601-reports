#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import json
import os
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

default_figsize = (15.1 / 2.54, 8.3 / 2.54)

def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)

def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 

def linear(x, a, b):
    return a * x + b


def job_colors():
    colors = [(55,126,184), (152,78,163), (77,175,74), (228,26,28)]

    with open('_build/colors.tex', 'w') as f:
        for name, color in zip(names, colors):
            f.write(r'\definecolor{{{}s}}{{rgb}}{{{},{},{}}}'.format(name, *[x/255 for x in color]) + '\n')

def bootstrap_time(T, show=False):
    time = []
    channel_val = []
    channel_err = []

    for i in range(1,7):
        print(i)
        time_raw = np.loadtxt('Data/prompt-{}.txt'.format(i))
        channel_time = time_raw[:,0]
        counts_time = time_raw[:,1]
        
        time_mean = []
        for a in range(100):
            boot_time = redraw_count(counts_time)
            popt, pconv = op.curve_fit(gauss, channel_time, boot_time, p0=[400+i*600, 200, 100])
            time_mean.append(popt[0])

            if show:
                x = np.linspace(0, 2000, 1000)
                y = gauss(x, *popt)
                pl.plot(channel_time[:2000], counts_time[:2000])
                pl.plot(x, y)
                pl.show()
                pl.clf()
        
        time_val, time_err = bootstrap.average_and_std_arrays(time_mean)
        print(time_val, '+-', time_err)
        time.append((i-1)*4)
        channel_val.append(time_val)
        channel_err.append(time_err)

    



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

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    options = parser.parse_args()

    bootstrap_time(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
