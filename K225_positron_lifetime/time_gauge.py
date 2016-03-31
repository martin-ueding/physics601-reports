# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

'''
Time gauge.
'''

import numpy as np

import bootstrap

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
