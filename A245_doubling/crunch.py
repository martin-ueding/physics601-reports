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

SAMPLES = 100


def linear(x, a, b):
    return a * x + b


def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 


def errorfunction(x, power, diam, x_offs):
    return power / 2 * sp.erfc(np.sqrt(8) / diam * (x - x_offs))


def cos_squared(x, ampl, x_offs, y_offs):
    return ampl * (np.cos(x - x_offs))**2 + y_offs


def job_power(T):
    data = np.loadtxt('Data/diode.tsv')
    norm_current = data[:, 0] * 1e-3
    norm_power_val = data[:, 1] * 1e-3
    norm_power_err = np.ones(norm_power_val.shape) * 1e-6
    norm_power_dist = bootstrap.make_dist(norm_power_val, norm_power_err)

    data = np.loadtxt('Data/diode_damped.tsv')
    damp_current = data[:, 0] * 1e-3
    damp_power_val = data[:, 1] * 1e-3
    damp_power_err = data[:, 2] * 1e-3
    damp_power_dist = bootstrap.make_dist(damp_power_val, damp_power_err)

    np.savetxt('_build/xy/diode_normal-data.tsv',
               np.column_stack([norm_current, norm_power_val, norm_power_err]))
    np.savetxt('_build/xy/diode_damped-data.tsv',
               np.column_stack([damp_current, damp_power_val, damp_power_err]))

    # Find the threshold current.
    sel = norm_power_val > 1e-3
    threshold_dist = []
    threshold_fit_x = np.linspace(0.05, 0.09, 100)
    threshold_fit_y_dist = []
    # Jackknife fit to find root.
    for i in range(len(norm_power_val[sel])):
        x = np.delete(norm_current[sel], i)
        y_val = np.delete(norm_power_val[sel], i)
        y_err = np.delete(norm_power_err[sel], i)
        popt, pconv = op.curve_fit(linear, x, y_val, sigma=y_err)
        a, b = popt
        root = -b / a
        threshold_dist.append(root)
        threshold_fit_y_dist.append(linear(threshold_fit_x, *popt))
    threshold_val, threshold_err = bootstrap.average_and_std_arrays(threshold_dist)
    threshold_fit_y_val, threshold_fit_y_err = bootstrap.average_and_std_arrays(threshold_fit_y_dist)

    T['threshold'] = siunitx(threshold_val, threshold_err)

    np.savetxt('_build/xy/diode_normal-band.tsv',
               bootstrap.pgfplots_error_band(threshold_fit_x, threshold_fit_y_val, threshold_fit_y_err))

    for norm_power, damp_power in zip(norm_power_dist, damp_power_dist):
        norm_inter = scipy.interpolate.interp1d(norm_current, norm_power)
        damp_inter = scipy.interpolate.interp1d(damp_current, damp_power)


def get_rayleigh_length(radius, wavelength, refractive_index, distance):
    numerator = refractive_index * radius**2 - np.sqrt(refractive_index**2 * radius**4 - 4 * wavelength**2 * distance**2)
    return np.pi * numerator / (2 * wavelength)


def get_waist(rayleigh_length, wavelength, refractive_index):
    return np.sqrt(rayleigh_length * wavelength / (refractive_index * np.pi))


def get_optimal_focal_length(beam_radius, refractive_index, wavelength, length):
    optimal_normalized_length = 2.84
    bracket = 2 * beam_radius**2 * refractive_index * np.pi * optimal_normalized_length / (length * wavelength) - 1
    factor = (length / (2 * optimal_normalized_length))**2
    return np.sqrt(bracket * factor)


def job_rayleigh_length(T):
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
    T['rayleigh_length_mm'] = siunitx(rayleigh_length_val / 1e-3, rayleigh_length_err / 1e-3, error_digits=2)

    waist_dist = list(itertools.filterfalse(np.isnan, [
        get_waist(rayleigh_length, wavelength, refractive_index)
        for rayleigh_length in rayleigh_length_dist
    ]))
    waist_val, waist_err = bootstrap.average_and_std_arrays(waist_dist)
    T['waist_mum'] = siunitx(waist_val / 1e-6, waist_err / 1e-6)

    normalized_length_dist = list([
        length / (2 * rayleigh_length)
        for rayleigh_length in rayleigh_length_dist
    ])
    normalized_length_val, normalized_length_err = bootstrap.average_and_std_arrays(normalized_length_dist)
    T['normalized_length'] = siunitx(normalized_length_val, normalized_length_err, error_digits=2)

    t = (normalized_length_val - 2.84) / normalized_length_err
    T['boyd_kleinman_ttest_t'] = siunitx(t)

    optimal_focal_length_dist = list([
        get_optimal_focal_length(beam_radius, refractive_index, wavelength, length)
        for beam_radius in beam_radius_dist
    ])
    optimal_focal_length_val, optimal_focal_length_err = bootstrap.average_and_std_arrays(optimal_focal_length_dist)
    T['optimal_focal_length_mm'] = siunitx(optimal_focal_length_val / 1e-3, optimal_focal_length_err / 1e-3, error_digits=2)


def make_lissajous(angle, ratio, offset, filename):
    x = np.sin(angle)
    y = np.sin(angle * ratio + offset)
    np.savetxt(filename, np.column_stack([x, y]))


def job_lissajous(T):
    angle = np.linspace(0, 8 * np.pi, 1000)

    make_lissajous(angle, 2, 0, '_build/xy/lissajous_2_0.tsv')
    make_lissajous(angle, 2, 0.2, '_build/xy/lissajous_2_02.tsv')
    make_lissajous(angle, 2, 1, '_build/xy/lissajous_2_1.tsv')

    make_lissajous(angle, 2.1, 0, '_build/xy/lissajous_21_0.tsv')
    make_lissajous(angle, 2.1, 0.7, '_build/xy/lissajous_21_07.tsv')
    make_lissajous(angle, 2.3, 2.4, '_build/xy/lissajous_23_24.tsv')

    make_lissajous(angle, 1, 0, '_build/xy/lissajous_1_0.tsv')
    make_lissajous(angle, 1, 1, '_build/xy/lissajous_1_1.tsv')
    make_lissajous(angle, 3, 0, '_build/xy/lissajous_3_0.tsv')


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

    job_power(T)
    job_lissajous(T)
    job_rayleigh_length(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

    pp = pprint.PrettyPrinter()
    print()
    print('Content in T dict:')
    pp.pprint(T)


if __name__ == "__main__":
    main()
