#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

import argparse
import pprint
import random

import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as op

from unitprint2 import siunitx
import bootstrap


def linear(x, a, b):
    return a * x + b


def do_resampling(T, prefix, x, y, y_err):
    fit_x = np.linspace(np.min(x), np.max(x), 100)
    y_dist = []
    popt_dist = []
    for i in range(40):
        boot_y = [random.gauss(Y, err) for Y, err in zip(y, y_err)]
        popt, pconv = op.curve_fit(linear, x, boot_y)
        boot_fit_y = linear(fit_x, *popt)

        y_dist.append(boot_fit_y)
        popt_dist.append(popt)

        np.savetxt('_build/xy/bootstrap-{}-{:d}-resampled.tsv'.format(prefix, i),
                   np.column_stack([x, boot_y]))
        np.savetxt('_build/xy/bootstrap-{}-{:d}-fit.tsv'.format(prefix, i),
                   np.column_stack([fit_x, boot_fit_y]))

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(y_dist)
    np.savetxt('_build/xy/bootstrap-{}-band.tsv'.format(prefix),
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    np.savetxt('_build/xy/bootstrap-{}-final-fit.tsv'.format(prefix),
               np.column_stack([fit_x, fit_y_val]))

    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    popt_val, pconv = op.curve_fit(linear, x, y, sigma=y_err)
    T['bootstrap_{}_popt'.format(prefix)] = siunitx(popt_val, popt_err, error_digits=2)
    T['bootstrap_{}_err'.format(prefix)] = siunitx(popt_err)


def do_jackknife(T, prefix, x, y, y_err):
    fit_x = np.linspace(np.min(x), np.max(x), 100)
    y_dist = []
    popt_dist = []
    for i in range(len(x)):
        jack_x = np.delete(x, i)
        jack_y = np.delete(y, i)

        popt, pconv = op.curve_fit(linear, jack_x, jack_y)
        boot_fit_y = linear(fit_x, *popt)

        y_dist.append(boot_fit_y)
        popt_dist.append(popt)

        np.savetxt('_build/xy/bootstrap-{}-{:d}-resampled.tsv'.format(prefix, i),
                   np.column_stack([jack_x, jack_y]))
        np.savetxt('_build/xy/bootstrap-{}-{:d}-fit.tsv'.format(prefix, i),
                   np.column_stack([fit_x, boot_fit_y]))

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(y_dist)
    np.savetxt('_build/xy/bootstrap-{}-band.tsv'.format(prefix),
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    np.savetxt('_build/xy/bootstrap-{}-final-fit.tsv'.format(prefix),
               np.column_stack([fit_x, fit_y_val]))


    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    popt_val, pconv = op.curve_fit(linear, x, y, sigma=y_err)
    T['bootstrap_{}_popt'.format(prefix)] = siunitx(popt_val, popt_err, error_digits=2)
    T['bootstrap_{}_err'.format(prefix)] = siunitx(popt_err)


def do_choice(T, prefix, x, y, y_err):
    fit_x = np.linspace(np.min(x), np.max(x), 100)
    y_dist = []
    popt_dist = []
    for i in range(len(x)):
        choice_x = bootstrap.generate_sample(x)
        choice_y = bootstrap.generate_sample(y)
        choice_y_err = bootstrap.generate_sample(y_err)

        popt, pconv = op.curve_fit(linear, choice_x, choice_y, sigma=choice_y_err)
        boot_fit_y = linear(fit_x, *popt)

        y_dist.append(boot_fit_y)
        popt_dist.append(popt)

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(y_dist)
    np.savetxt('_build/xy/bootstrap-{}-band.tsv'.format(prefix),
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    np.savetxt('_build/xy/bootstrap-{}-final-fit.tsv'.format(prefix),
               np.column_stack([fit_x, fit_y_val]))

    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    popt_val, pconv = op.curve_fit(linear, x, y, sigma=y_err)
    T['bootstrap_{}_popt'.format(prefix)] = siunitx(popt_val, popt_err, error_digits=2)
    T['bootstrap_{}_err'.format(prefix)] = siunitx(popt_err)


def do_pconv(T, prefix, x, y, y_err):
    popt, pconv = op.curve_fit(linear, x, y, sigma=y_err)

    fit_x = np.linspace(np.min(x), np.max(x), 100)
    fit_y = linear(fit_x, *popt)

    np.savetxt('_build/xy/bootstrap-{}-normal-data.tsv'.format(prefix),
               np.column_stack([x, y, y_err]))
    np.savetxt('_build/xy/bootstrap-{}-normal-fit.tsv'.format(prefix),
               np.column_stack([fit_x, fit_y]))

    T['bootstrap_{}_popt'.format(prefix)] = siunitx(popt, np.sqrt(pconv.diagonal()), error_digits=2)
    T['bootstrap_{}_val'.format(prefix)] = siunitx(popt)
    T['bootstrap_{}_err'.format(prefix)] = siunitx(np.sqrt(pconv.diagonal()))


def generate_normal_chi_sq(num_points):
    x = np.linspace(1, 7, num_points)
    y_err = np.ones(x.shape)
    y = [random.gauss(X, err) for X, err in zip(x, y_err)]
    return x, y, y_err


def main(T={}):
    options = _parse_args()
    pp = pprint.PrettyPrinter()

    NUM_POINTS = 7

    x, y, y_err = generate_normal_chi_sq(NUM_POINTS)
    do_pconv(T, 'normal_pconv', x, y, y_err)
    do_resampling(T, 'normal_resampling', x, y, y_err)
    do_jackknife(T, 'normal_jackknife', x, y, y_err)
    do_choice(T, 'normal_choice', x, y, y_err)

    do_pconv(T, 'small_pconv', x, y, y_err*10)
    do_resampling(T, 'small_resampling', x, y, y_err*10)
    do_jackknife(T, 'small_jackknife', x, y, y_err*10)
    do_choice(T, 'small_choice', x, y, y_err*10)

    do_pconv(T, 'large_pconv', x, y, y_err/10)
    do_resampling(T, 'large_resampling', x, y, y_err/10)
    do_jackknife(T, 'large_jackknife', x, y, y_err/10)
    do_choice(T, 'large_choice', x, y, y_err/10)

    pp.pprint(T)


def _parse_args():
    '''
    Parses the command line arguments.

    :return: Namespace with arguments.
    :rtype: Namespace
    '''
    parser = argparse.ArgumentParser(description='')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
