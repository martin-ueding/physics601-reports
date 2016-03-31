# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

'''
Functions for the expected lifetime spectrum.
'''

import glob
import itertools

import numpy as np
import scipy.optimize as op

import bootstrap
import conf
import models


def job_lifetime_spectra(T):
    files = glob.glob('Data/in-*.txt')

    x = np.linspace(1000, 3000, 500)

    for i, file_ in zip(itertools.count(), files):
        print('Working on lifetime spectrum', file_)

        data = np.loadtxt(file_)
        channel = data[:, 0]
        counts = data[:, 1]

        results = []
        for a in range(conf.SAMPLES):
            boot_counts = bootstrap.redraw_count(counts)
            results.append(_lifetime_kernel(channel, boot_counts, x))
        popt_dist, y_dist = zip(*results)

        popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
        mean_val, width_val, A_0_val, A_t_val, tau_0_val, tau_t_val, bg_val = popt_val
        mean_err, width_err, A_0_err, A_t_err, tau_0_err, tau_t_err, bg_err = popt_err

        y_val, y_err = bootstrap.average_and_std_arrays(y_dist)

        np.savetxt('_build/xy/lifetime-{}-data.tsv'.format(i),
                   np.column_stack((channel, counts)))
        np.savetxt('_build/xy/lifetime-{}-fit.tsv'.format(i),
                   np.column_stack((x, y_val)))
        np.savetxt('_build/xy/lifetime-{}-band.tsv'.format(i),
                   bootstrap.pgfplots_error_band(x, y_val, y_err))



def _lifetime_kernel(channel, counts, x):
    p0 = [1600, 45, 180, 180, 40, 40, 0]
    popt, pconv = op.curve_fit(models.lifetime_spectrum, channel, counts, p0=p0)
    y = models.lifetime_spectrum(x, *popt)
    return popt, y
