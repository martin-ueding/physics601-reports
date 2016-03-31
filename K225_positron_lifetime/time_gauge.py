# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

'''
Time gauge.
'''

import numpy as np

import bootstrap

import models

SAMPLES = 300


def job_time_gauge(T):
    channels, all_counts = _get_raw_data()

    _write_total_counts(channels, all_counts)
    _write_long_term(channels, all_counts[-1])


def _get_raw_data():
    '''
    :returns tuple(array, list(array)): Channels and a list with all seven
    prompt curves.
    '''
    all_counts = []

    for i in range(1,7):
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
    total_counts = _get_total_counts(all_counts)

    error_band_1 = bootstrap.pgfplots_error_band(channels[500:3500],
                                                 counts_tot[500:3500],
                                                 np.sqrt(counts_tot[500:3500]))

    np.savetxt('_build/xy/prompts-short.txt', error_band_1)


def _write_long_term(channels, counts):
    np.savetxt('_build/xy/prompts-long.txt', error_band_2)
    error_band_2 = bootstrap.pgfplots_error_band(channels[3600:4200],
                                                 counts[3600:4200],
                                                 np.sqrt(counts[3600:4200]))


def _fit_prompt(channels, counts, idx):
    results = []
    x = np.linspace(0, 8000, 8000)
    for sample in range(SAMPLES):
        boot_counts = redraw_count(counts)
        results.append(_fit_prompt_kernel(channels, boot_counts, idx, x))
    
    (mean_dist, width_dist, amp_dist), y_dist = zip(*results)

    mean_val, mean_err = bootstrap.average_and_std_arrays(mean_dist)
    width_val, width_err = bootstrap.average_and_std_arrays(width_dist)
    amplitude_val, amplitude_err = bootstrap.average_and_std_arrays(amplitude_dist)
    y_val, y_err = bootstrap.average_and_std_arrays(y_dist)

    # Create files for prompt curve fits.
    lower = int(mean_val - 3 * width_val)
    upper = int(mean_val + 3 * width_val)
    np.savetxt('_build/xy/prompt-{}-fit.txt'.format(i),
               np.column_stack((x[lower:upper], y_val[lower:upper])))
    np.savetxt('_build/xy/prompt-{}-band.txt'.format(i),
               bootstrap.pgfplots_error_band(x[lower:upper],
                                             y_val[lower:upper],
                                             y_err[lower:upper]))

    time = (i - 1) * 4

    return time, channel_val, channel_err


def _fit_prompt_kernel(channels, counts, idx, x):
    p0 = [400 + i * 600, 200, 100]
    popt, pconv = op.curve_fit(models.gauss, channels, counts, p0=p0)

    y = models.gauss(x, mean_val, width_val, amplitude_val)

    return popt, y
