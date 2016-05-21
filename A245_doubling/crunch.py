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
import scipy.special as sp
import mpl_toolkits.mplot3d.axes3d as p3

from unitprint2 import siunitx
import bootstrap


def linear(x, a, b):
    return a * x + b


def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 


def errorfunction(x, power, diam, x_offs):
    return power / 2 * sp.erfc(np.sqrt(8) / diam * (x - x_offs))


def cos_squared(x, ampl, x_offs, y_offs):
    return ampl * (np.cos(x - x_offs))**2 + y_offs


def get_rayleigh_length(radius, wavelength, refractive_index, distance):
    numerator = refractive_index * radius**2 - np.sqrt(refractive_index**2 * radius**4 - 4 * wavelength**2 * distance**2)
    return np.pi * numerator / (2 * wavelength)


def get_waist(rayleigh_length, wavelength, refractive_index):
    return np.sqrt(rayleigh_length * wavelength / (refractive_index * np.pi))


def task_rayleigh_length(T):
    beam_diameter_val = 3.5e-3
    beam_diameter_err = 0.5e-3
    refractive_index = 2.2
    wavelength = 987e-9
    length = 5e-3
    distance = 60e-3

    beam_radius_val = beam_diameter_val / 2
    beam_radius_err = beam_diameter_err / 2

    T['beam_radius'] = siunitx(beam_radius_val, beam_radius_err)

    beam_radius_dist = bootstrap.make_dist(beam_radius_val, beam_diameter_err)

    rayleigh_length_dist = list(itertools.filterfalse(np.isnan, [
        get_rayleigh_length(beam_radius, wavelength, refractive_index, distance)
        for beam_radius in beam_radius_dist
    ]))
    rayleigh_length_val, rayleigh_length_err = bootstrap.average_and_std_arrays(rayleigh_length_dist)
    T['rayleigh_length'] = siunitx(rayleigh_length_val, rayleigh_length_err)
    print(rayleigh_length_dist)

    waist_dist = list(itertools.filterfalse(np.isnan, [
        get_waist(rayleigh_length, wavelength, refractive_index)
        for rayleigh_length in rayleigh_length_dist
    ]))
    waist_val, waist_err = bootstrap.average_and_std_arrays(waist_dist)
    T['waist'] = siunitx(waist_val, waist_err)
    print(waist_dist)

    normalized_length_dist = list([
        length / (2 * rayleigh_length)
        for rayleigh_length in rayleigh_length_dist
    ])
    normalized_length_val, normalized_length_err = bootstrap.average_and_std_arrays(normalized_length_dist)
    T['normalized_length'] = siunitx(normalized_length_val, normalized_length_err)
    print(normalized_length_dist)


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

    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    task_rayleigh_length(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

    pp = pprint.PrettyPrinter()
    pp.pprint(T)


if __name__ == "__main__":
    main()
