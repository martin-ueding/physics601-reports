# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

'''
Time gauge.
'''

import itertools

import numpy as np
import scipy.optimize as op

from unitprint2 import siunitx
import bootstrap

import models

SAMPLES = 10


def job_time_gauge(T):
    channels, all_counts = _get_raw_data()

    _write_total_counts(channels, all_counts)
    _write_long_term(channels, all_counts[-1])

    results = []
    for idx, counts in zip(itertools.count(1), all_counts):
        results.append(_fit_prompt(channels, counts, idx))
    time, channel_val, channel_err = zip(*results)

    channel_val = np.array(channel_val)
    channel_err = np.array(channel_err)
    time = np.array(time)

    T['time_gauge_param'] = list(zip(
        siunitx(time),
        siunitx(channel_val, channel_err)
    ))

    _get_slope_and_intercept(time, channel_val, channel_err)


def _get_slope_and_intercept(time, channel_val, channel_err):
    slope_dist = []
    intercept_dist = []
    y_dist = []

    x = np.linspace(np.min(channel_val), np.max(channel_val), 100)

    for i in range(len(channel_val)):
        channel_jackknife = np.delete(channel_val, i)
        time_jackknife = np.delete(time, i)
        
        popt, pconv = op.curve_fit(models.linear, channel_jackknife, time_jackknife)
        
        slope_dist.append(popt[0])
        intercept_dist.append(popt[1])

        y = models.linear(x, *popt)
        y_dist.append(y)

    slope_val, slope_err = bootstrap.average_and_std_arrays(slope_dist)
    intercept_val, intercept_err = bootstrap.average_and_std_arrays(intercept_dist)
    y_val, y_err = bootstrap.average_and_std_arrays(y_dist)

    np.savetxt('_build/xy/time-gauge-fit.tsv',
               np.column_stack((x, y_val)))
    np.savetxt('_build/xy/time-gauge-band.tsv',
               bootstrap.pgfplots_error_band(x, y_val, y_err))


    return slope_val, slope_err, intercept_val, intercept_err


def _get_raw_data():
    '''
    :returns tuple(array, list(array)): Channels and a list with all seven
    prompt curves.
    '''
    all_counts = []

    for i in range(1, 7):
        data = np.loadtxt('Data/prompt-{}.txt'.format(i))
        channels = data[:,0]
        counts = data[:,1]
        all_counts.append(counts)

    return channels, all_counts


def _get_total_counts(all_counts):
    return np.sum(all_counts, axis=0)


def _write_total_counts(channels, all_counts):
    '''
    Write files for prompt curve plotting.
    '''
    counts_tot = _get_total_counts(all_counts)

    error_band_1 = bootstrap.pgfplots_error_band(channels[500:3500],
                                                 counts_tot[500:3500],
                                                 np.sqrt(counts_tot[500:3500]))

    np.savetxt('_build/xy/prompts-short.tsv', error_band_1)


def _write_long_term(channels, counts):
    error_band_2 = bootstrap.pgfplots_error_band(channels[3600:4200],
                                                 counts[3600:4200],
                                                 np.sqrt(counts[3600:4200]))
    np.savetxt('_build/xy/prompts-long.tsv', error_band_2)


def _fit_prompt(channels, counts, idx):
    results = []
    x = np.linspace(0, 8000, 3000)
    for sample in range(SAMPLES):
        boot_counts = bootstrap.redraw_count(counts)
        results.append(_fit_prompt_kernel(channels, boot_counts, idx, x))
    
    popt_dist, y_dist = zip(*results)

    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    mean_val, width_val, amplitude_val = popt_val
    mean_err, width_err, amplitude_err = popt_err

    y_val, y_err = bootstrap.average_and_std_arrays(y_dist)

    # Create files for prompt curve fits.
    sel = ((mean_val - 4 * width_val) < x) & (x < (mean_val + 4 * width_val))
    np.savetxt('_build/xy/prompt-{}-fit.tsv'.format(idx),
               np.column_stack((x[sel], y_val[sel])))
    np.savetxt('_build/xy/prompt-{}-band.tsv'.format(idx),
               bootstrap.pgfplots_error_band(x[sel],
                                             y_val[sel],
                                             y_err[sel]))

    print(popt_val)

    time = (idx - 1) * 4

    return time, mean_val, mean_err


def _fit_prompt_kernel(channels, counts, idx, x):
    p0 = [400 + idx * 600, 45, 15000]
    popt, pconv = op.curve_fit(models.gauss, channels, counts, p0=p0)

    y = models.gauss(x, *popt)

    return popt, y
