#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import json
import os
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


def job_colors():
    colors = [(55,126,184), (152,78,163), (77,175,74), (228,26,28)]

    with open('_build/colors.tex', 'w') as f:
        for name, color in zip(names, colors):
            f.write(r'\definecolor{{{}s}}{{rgb}}{{{},{},{}}}'.format(name, *[x/255 for x in color]) + '\n')




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

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
