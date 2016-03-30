#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as pl
import numpy as np
import scipy.interpolate
import scipy.misc
import scipy.ndimage.filters
import scipy.optimize as op
import scipy.stats
import mpl_toolkits.mplot3d.axes3d as p3

from unitprint2 import siunitx
import bootstrap

default_figsize = (15.1 / 2.54, 8.3 / 2.54)

def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)

def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 

def linear(x, a, b):
    return a * x + b

def prepare_for_pgf(file, lower=0, upper=8000, error=False):
    data = np.loadtxt('Data/{}.txt'.format(file))
    channel = data[:,0]
    counts = data[:,1]

    if error:
        np.savetxt('_build/xy/{}.txt'.format(file), bootstrap.pgfplots_error_band(channel[lower:upper], counts_tot[lower:upper], np.sqrt(counts_tot[lower:upper])))
    else:
        np.savetxt('_build/xy/{}.txt'.format(file), channel[lower:upper], counts_tot[lower:upper])


def job_colors():
    colors = [(55,126,184), (152,78,163), (77,175,74), (228,26,28)]

    with open('_build/colors.tex', 'w') as f:
        for name, color in zip(names, colors):
            f.write(r'\definecolor{{{}s}}{{rgb}}{{{},{},{}}}'.format(name, *[x/255 for x in color]) + '\n')

def time_gauge(T, show_gauss=False, show_lin=False):
    time = []
    channel_val = []
    channel_err = []

    # go through all six prompt-files
    for i in range(1,7):
        time_raw = np.loadtxt('Data/prompt-{}.txt'.format(i))
        channel = time_raw[:,0]
        counts = time_raw[:,1]
        if i==1:
            counts_tot = counts
        elif i<6:
            counts_tot += counts
        
         # bootstrap:
         # - draw new counts from gaussian distribution with width of 'sqrt(N)'
         # - fit gaussian distribution to drawn data
         # - add mean to array
        mean = []
        width = []
        amplitude = []
        for a in range(2):
            boot_counts = redraw_count(counts)
            popt, pconv = op.curve_fit(gauss, channel, boot_counts, p0=[400+i*600, 200, 100])
            mean.append(popt[0])
            width.append(popt[1])
            amplitude.append(popt[2])

        # find average and standard deviation in arrays
        mean_val, mean_err = bootstrap.average_and_std_arrays(mean)
        width_val, width_err = bootstrap.average_and_std_arrays(width)
        amplitude_val, amplitude_err = bootstrap.average_and_std_arrays(amplitude)

        # create files for prompt curve fits
        x = np.linspace(mean_val-200, mean_val+200, 100)
        y = gauss(x, mean_val, width_val, amplitude_val)

        np.savetxt('_build/xy/prompt-{}-fit.txt'.format(i), np.column_stack([x, y]))

        # write result into new channel arrays
        channel_val.append(mean_val)
        channel_err.append(mean_err)

        # write real time for gauging
        time.append((i-1)*4)

    # write files for prompt curve plotting
    np.savetxt('_build/xy/prompts-short.txt', bootstrap.pgfplots_error_band(channel[500:3500], counts_tot[500:3500], np.sqrt(counts_tot[500:3500])))
    np.savetxt('_build/xy/prompts-long.txt', bootstrap.pgfplots_error_band(channel[3600:4200], counts[3600:4200], np.sqrt(counts[3600:4200])))

    # convert lists to arrays
    channel_val = np.array(channel_val)
    channel_err = np.array(channel_err)
    time = np.array(time)

    T['time_gauge_param'] = list(zip(*[
        map(str, time),
        siunitx(channel_val, channel_err)
    ]))

    # linear fit with delete-1-jackknife
    slope = []
    intercept = []
    for i in range(len(channel_val)):
        channel_jackknife = np.delete(channel_val, i)
        time_jackknife = np.delete(time, i)
        
        popt, pconv = op.curve_fit(linear, channel_jackknife, time_jackknife)
        
        slope.append(popt[0])
        intercept.append(popt[1])

        if show_lin:
            x = np.linspace(0, 4000, 1000)
            y = linear(x, *popt)
            pl.plot(channel_val, time, linestyle="none", marker="o")
            pl.plot(x, y)
            pl.show()
            pl.clf()

    slope_val, slope_err = bootstrap.average_and_std_arrays(slope)
    intercept_val, intercept_err = bootstrap.average_and_std_arrays(intercept)

    # files for fit and plot of time gauge 
    x = np.linspace(750, 4000, 100)
    y = linear(x, slope_val, intercept_val)

    np.savetxt('_build/xy/time_gauge_plot.txt', np.column_stack([channel_val,time , channel_err]))
    np.savetxt('_build/xy/time_gauge_fit.txt', np.column_stack([x,y]))
        

    T['time_gauge_slope'] = siunitx(slope_val*1e3, slope_err*1e3)
    T['time_gauge_intercept'] = siunitx(intercept_val, intercept_err)

    # time resolution

    T['width_6'] = siunitx(width_val , width_err)
    FWHM_val = 2*np.sqrt(2*np.log(2)) * width_val 
    FWHM_err = 2*np.sqrt(2*np.log(2)) * width_err 
    T['FWHM_6'] = siunitx(FWHM_val , FWHM_err)
    
    time_res = FWHM_val * slope_val
    time_res_err = np.sqrt((FWHM_val * slope_err)**2 + (FWHM_err * slope_val)**2)
    T['time_resolution'] = siunitx(time_res , time_res_err)


def redraw_count(a):
    '''
    Takes a ``np.array`` with counts and re-draws the counts from the implicit
    Gaussian distribution with width ``sqrt(N)``.
    '''
    out = [random.gauss(x, np.sqrt(x)) for x in a]
    return np.array(out).reshape(a.shape)


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    options = parser.parse_args()

    time_gauge(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
