#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

import argparse
import random

import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as op

from unitprint2 import siunitx
import bootstrap


def linear(x, a, b):
    return a * x + b


def main(T):
    options = _parse_args()

    NUM_POINTS = 7

    # x = np.linspace(1, 5, NUM_POINTS)
    # errs = [np.abs(random.gauss(1, 0.5)) for X in x]
    # y = [random.gauss(X, err) for X, err in zip(x, errs)]

    # np.savetxt('example.txt',
    #            np.column_stack([x, y, errs]))

    data = np.loadtxt('example.txt')
    x = data[:, 0]
    y = data[:, 1]
    y_err = data[:, 2]

    popt, pconv = op.curve_fit(linear, x, y, sigma=y_err)

    fit_x = np.linspace(np.min(x), np.max(x), 100)
    fit_y = linear(fit_x, *popt)

    np.savetxt('_build/xy/bootstrap-normal-data.tsv',
               np.column_stack([x, y, y_err]))
    np.savetxt('_build/xy/bootstrap-normal-fit.tsv',
               np.column_stack([fit_x, fit_y]))

    T['pconv_example_popt'] = siunitx(popt, np.sqrt(pconv.diagonal()), error_digits=2)

    y_dist = []
    popt_dist = []
    for i in range(40):
        boot_y = [random.gauss(Y, err) for Y, err in zip(y, y_err)]
        popt, pconv = op.curve_fit(linear, x, boot_y)
        boot_fit_y = linear(fit_x, *popt)

        y_dist.append(boot_fit_y)
        popt_dist.append(popt)

        np.savetxt('_build/xy/bootstrap-{:d}-resampled.tsv'.format(i),
                   np.column_stack([x, boot_y]))
        np.savetxt('_build/xy/bootstrap-{:d}-fit.tsv'.format(i),
                   np.column_stack([fit_x, boot_fit_y]))

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(y_dist)
    np.savetxt('_build/xy/bootstrap-band.tsv',
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    np.savetxt('_build/xy/bootstrap-final-fit.tsv'.format(i),
               np.column_stack([fit_x, fit_y_val]))

    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    T['bootstrap_example_popt'] = siunitx(popt_val, popt_err, error_digits=2)

    y_dist = []
    popt_dist = []
    for i in range(len(x)):
        jack_x = np.delete(x, i)
        jack_y = np.delete(y, i)

        popt, pconv = op.curve_fit(linear, jack_x, jack_y)
        boot_fit_y = linear(fit_x, *popt)

        y_dist.append(boot_fit_y)
        popt_dist.append(popt)

        np.savetxt('_build/xy/jackknife-{:d}-resampled.tsv'.format(i),
                   np.column_stack([jack_x, jack_y]))
        np.savetxt('_build/xy/jackknife-{:d}-fit.tsv'.format(i),
                   np.column_stack([fit_x, boot_fit_y]))

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(y_dist)
    np.savetxt('_build/xy/jackknife-band.tsv',
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    np.savetxt('_build/xy/jackknife-final-fit.tsv'.format(i),
               np.column_stack([fit_x, fit_y_val]))

    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    T['jackknife_example_popt'] = siunitx(popt_val, popt_err, error_digits=2)

    y_dist = []
    popt_dist = []
    for j in range(30):
        boot_y = [random.gauss(Y, err) for Y, err in zip(y, y_err)]
        for i in range(len(x)):
            jack_x = np.delete(x, i)
            jack_y = np.delete(boot_y, i)

            popt, pconv = op.curve_fit(linear, jack_x, jack_y)
            boot_fit_y = linear(fit_x, *popt)

            y_dist.append(boot_fit_y)
            popt_dist.append(popt)

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(y_dist)
    np.savetxt('_build/xy/combined-band.tsv',
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    np.savetxt('_build/xy/combined-final-fit.tsv'.format(i),
               np.column_stack([fit_x, fit_y_val]))

    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    T['combined_example_popt'] = siunitx(popt_val, popt_err, error_digits=2)




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
