# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

'''
Functions for the expected lifetime spectrum.
'''

import glob
import itertools
import os
import re

import numpy as np
import scipy.optimize as op

import bootstrap
import conf
import models


TEMP_PATTERN = re.compile('in-(\d+(?:,\d+)?)-(\d+(?:,\d+)?)C\.txt')
p0 = [10.5, 0.3, 210, 190, 0.07, 0.8, 0]


def indium_lifetime_spectra(T, slope_val):
    files = glob.glob('Data/in-*.txt')

    return_values = []

    for i, file_ in zip(itertools.count(), sorted(files)):
        print('Working on lifetime spectrum', file_)

        temp_lower, temp_upper = get_temp(file_)

        data = np.loadtxt(file_)
        channel = data[:, 0]
        counts = data[:, 1]
        counts_err = np.sqrt(counts)

        counts_err[counts_err == 0] = 1

        time = channel * slope_val

        np.savetxt('_build/xy/lifetime-{:04d}-data.tsv'.format(int(temp_lower*10)),
                   np.column_stack((time, counts)))

        x = np.linspace(7, 20, 500)

        results = []
        for a in range(conf.SAMPLES):
            boot_counts = bootstrap.redraw_count(counts)
            results.append(_lifetime_kernel(time, boot_counts, counts_err, x))
        popt_dist, y_dist = zip(*results)

        popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
        mean_val, width_val, A_0_val, A_t_val, tau_0_val, tau_t_val, bg_val = popt_val
        mean_err, width_err, A_0_err, A_t_err, tau_0_err, tau_t_err, bg_err = popt_err

        print(popt_val)

        y_val, y_err = bootstrap.average_and_std_arrays(y_dist)

        np.savetxt('_build/xy/lifetime-{:04d}-fit.tsv'.format(int(temp_lower*10)),
                   np.column_stack((x, y_val)))
        np.savetxt('_build/xy/lifetime-{:04d}-band.tsv'.format(int(temp_lower*10)),
                   bootstrap.pgfplots_error_band(x, y_val, y_err))

        return_values.append([temp_lower, temp_upper, tau_0_val, tau_0_err,
                              tau_t_val, tau_t_err])

    return zip(*return_values)


def _lifetime_kernel(time, counts, counts_err, x):
    popt, pconv = op.curve_fit(models.lifetime_spectrum, time, counts, sigma=counts_err, p0=p0)
    y = models.lifetime_spectrum(x, *popt)
    return popt, y


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

    raise RuntimeError('Filename {} could not be parsed for temperature.'.format(filename))


if __name__ == '__main__':
    x = np.linspace(0, 50, 1000)
    y = models.lifetime_spectrum(x, *p0)
    np.savetxt('_build/xy/lifetime-test.tsv', np.column_stack((x, y)))
