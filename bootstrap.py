#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014-2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

from __future__ import division, absolute_import, print_function, \
    unicode_literals

import random

import matplotlib.pyplot as pl
import numpy as np


def save_hist(dist, filename):
    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(dist)
    fig.tight_layout()
    fig.savefig(filename)


def make_dist(val, err, n=50):
    # TODO Make this work nicer with arrays.
    if isinstance(val, (float, int)):
        dist = [random.gauss(val, err) for i in range(n)]
    elif isinstance(val, (np.ndarray, list)):
        dist = []
        for v, e in zip(val, err):
            dist.append([random.gauss(v, e) for i in range(n)])
        dist = zip(*dist)

    else:
        raise RuntimeError('Unsupported type for make_dist')
    return dist


def redraw_count(a):
    '''
    Takes a ``np.array`` with counts and re-draws the counts from the implicit
    Gaussian distribution with width ``sqrt(N)``.
    '''
    out = [random.gauss(x, np.sqrt(x)) for x in a]
    return np.array(out).reshape(a.shape)


def pgfplots_error_band(x, y_val, y_err):
    return np.column_stack([
        np.concatenate((x, x[::-1])),
        np.concatenate((
            (y_val-y_err),
            (y_val+y_err)[::-1]
        ))
    ])


def average_arrays(arrays):
    '''
    Computes the element wise average of a list of arrays.
    '''
    total = np.column_stack(arrays)

    val = np.mean(total, axis=1)

    return val


def average_and_std_arrays(arrays):
    '''
    Computes the element wise average and standard deviation of a list of
    arrays.
    '''
    total = np.array(arrays)
    
    val = np.mean(total, axis=0)
    err = np.std(total, axis=0)

    return val, err


def percentile_arrays(arrays, value=None, interval=68.3):
    total = np.array(arrays)

    r_up = 50 + interval/2
    r_down = 50 - interval/2

    print('Percentiles:', r_up, r_down)

    p_up = np.percentile(arrays, r_up, axis=0)
    p_down = np.percentile(arrays, r_down, axis=0)

    if value is None:
        value = np.median(arrays, axis=0)

    e_up = p_up - value 
    e_down = value - p_down

    return e_up, e_down


def std_arrays(arrays):
    '''
    Computes the element wise standard deviation of a list of arrays.
    '''
    total = np.array(arrays)
    
    err = np.std(total, axis=0)

    return err


def bootstrap_and_transform(transform, sets, sample_count=250, seed=None):
    '''
    Bootstraps the sets and transforms them.

    This is the recommended method!

    The return value of the function is assumed to be a one dimensional NumPy
    array. The return value of this function is one array with the values and
    another with the errors.
    '''
    random.seed(seed)

    results = []
    for sample_id in range(sample_count):
        sample = generate_sample(sets)
        try:
            transformed = transform(sample)
        except RuntimeError as e:
            print(e)
        else:
            results.append(transformed)

    val, err = average_and_std_arrays(results)

    return val, err


def generate_reduced_samples(sets, sample_count, seed=None):
    '''
    Generates all bootstrap samples at once.

    For the analysis, the correlation matrix needs to be computed from the
    bootstrap samples. Therefore, all the bootstrap samples are needed at the
    same time and :py:`bootstrap_and_transform` cannot be used here.

    :returns: List of averaged bootstrap samples
    :rtype: list(sets)
    '''
    random.seed(seed)

    results = [
        average_combined_array(generate_sample(sets))
        for sample_if in range(sample_count)
    ]

    return results


def generate_sample(elements):
    '''
    Generates a sample from the given list.

    The number of elements in the sample is taken to be the same as the number
    of elements given.
    '''
    result = []
    for i in range(len(elements)):
        result.append(random.choice(elements))

    return result


def average_combined_array(combined):
    '''
    Given a list of tuples or arrays of the kind “configuration → n-point →
    correlator“, it creates the average of the correlator over all the
    configurations for each “n-point”.

    It will then return the averaged out correlator for the two- and for the
    four-point correlation function.

    The input list is converted into the structure “n-point → configuration →
    correlator“ using zip(). Then the function average_arrays() is used on each
    element of the outer list.
    '''
    result = []
    npoints = zip(*combined)
    for npoint in npoints:
        result.append(average_and_std_arrays(npoint))

    return result
