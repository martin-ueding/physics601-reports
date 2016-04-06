#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import glob
import json
import models
import os
import pickle
import random
import re
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

default_figsize = (15.1 / 2.54, 8.3 / 2.54)

TEMP_PATTERN = re.compile('in-(\d+(?:,\d+)?)-(\d+(?:,\d+)?)C\.txt')
BOOTSTRAP_SAMPLES = 10



def dandify_plot():
    pl.legend(loc='best')
    pl.grid(True)
    pl.margins(0.05)
    pl.tight_layout()


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
        first = float(m.group(1).replace(',', '.')) + 273.15
        second = float(m.group(2).replace(',', '.')) + 273.15

        return (first, second)

    return None


def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)


def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 


def linear(x, a, b):
    return a * x + b


def exp_decay(x, a, b):
    return a * np.exp(- b * x)


def prepare_for_pgf(filename,  error=False, show=False):
    data = np.loadtxt('Data/{}.txt'.format(filename))
    channel = data[:,0]
    counts = data[:,1]

    lower = 0
    upper = 8000
    sieve_factor = 10

    if error:
        np.savetxt('_build/xy/{}.txt'.format(filename), bootstrap.pgfplots_error_band(channel[lower:upper:sieve_factor], counts[lower:upper:sieve_factor], np.sqrt(counts[lower:upper:sieve_factor])))
    else:
        np.savetxt('_build/xy/{}.txt'.format(filename), np.column_stack([channel[lower:upper:sieve_factor], counts[lower:upper:sieve_factor]]))

    if show:
        pl.plot(channel, counts, linestyle="none", marker="o")
        pl.show()
        pl.clf()


def prepare_files(T):
    prepare_for_pgf('lyso-li', error=True, show=False)
    prepare_for_pgf('lyso-re', error=True, show=False)
    prepare_for_pgf('na-li', error=True, show=False)
    prepare_for_pgf('na-re', error=True, show=False)
    prepare_for_pgf('na-511-re', show=False)
    prepare_for_pgf('na-511-li', show=False)
    prepare_for_pgf('na-1275-li', show=False)


def job_colors():
    colors = [(55,126,184), (152,78,163), (77,175,74), (228,26,28)]

    with open('_build/colors.tex', 'w') as f:
        for name, color in zip(names, colors):
            f.write(r'\definecolor{{{}s}}{{rgb}}{{{},{},{}}}'.format(name, *[x/255 for x in color]) + '\n')


def lifetime(T):
    slope, width = time_gauge(T)
    get_acryl_data(T, slope, width)
    sys.exit(200)
    get_indium_data(T, slope, width)


def get_acryl_data(T, slope_val, width):
    data = np.loadtxt('Data/longlong.txt')
    channel = data[:, 0]
    time = slope_val * channel
    counts = data[:, 1]

    x = np.linspace(np.min(time), np.max(time), 2000)

    fit_func = lambda t, mean, A_0, A_t, tau_0, tau_t, BG: \
            models.lifetime_spectrum(t, mean, width, A_0, A_t, tau_0, tau_t, BG)

    sel = (10 < time) & (time < 50)

    results = []

    for sample_id in range(BOOTSTRAP_SAMPLES):
        print('Bootstrap sample', sample_id, 'running …')

        boot_counts = bootstrap.redraw_count(counts)

        p0 = [10.5, 5e3, 9e3, 2.17, 0.508, 0]
        popt, pconv = op.curve_fit(fit_func, time[sel], boot_counts[sel], p0=p0)
        mean, A_0, A_t, tau_0, tau_t, BG = popt

        intens_0 = A_0 / (A_0 + A_t)
        intens_t = A_t / (A_0 + A_t)
        tau_bar = intens_0 * tau_0 + intens_t * tau_t
        y = fit_func(x, *popt)
        tau_f = 1 / (intens_0 / tau_0 - intens_t / tau_t)
        sigma_c = 1 / tau_0 - 1 / tau_f

        sel1 = (10.92 < time) & (time < 11.58)
        sel2 = (13.11 < time) & (time < 22)

        sels = [sel1, sel2]
        lin_results = []
        for sel_lin in sels:
            popt_lin, pconv_lin = op.curve_fit(exp_decay, time[sel_lin], boot_counts[sel_lin], p0=[1e5, 0.3])
            y_lin = exp_decay(x, *popt_lin)

            lin_results.append(y_lin)
            lin_results.append(popt_lin)


        results.append([
            tau_0,
            tau_bar,
            tau_f,
            tau_t,
            intens_0,
            intens_t,
            y,
            popt,
            sigma_c,
        ] + lin_results)
        
    tau_0_dist, tau_bar_dist, tau_f_dist, tau_t_dist, intens_0_dist, \
            intens_t_dist, lifetime_y_dist, lifetime_popt_dist, sigma_c_dist, \
            y_lin1_dist, popt_lin1_dist, y_lin2_dist, popt_lin2_dist \
            = zip(*results)

    tau_0_val, tau_0_err = bootstrap.average_and_std_arrays(tau_0_dist)
    tau_t_val, tau_t_err = bootstrap.average_and_std_arrays(tau_t_dist)
    tau_f_val, tau_f_err = bootstrap.average_and_std_arrays(tau_f_dist)
    tau_bar_val, tau_bar_err = bootstrap.average_and_std_arrays(tau_bar_dist)

    popt_val, popt_err = bootstrap.average_and_std_arrays(lifetime_popt_dist)
    y_val, y_err = bootstrap.average_and_std_arrays(lifetime_y_dist)

    popt_lin1_val, popt_lin1_err = bootstrap.average_and_std_arrays(popt_lin1_dist)
    y_lin1_val, y_lin1_err = bootstrap.average_and_std_arrays(y_lin1_dist)
    popt_lin2_val, popt_lin2_err = bootstrap.average_and_std_arrays(popt_lin2_dist)
    y_lin2_val, y_lin2_err = bootstrap.average_and_std_arrays(y_lin2_dist)

    print('tau_0', siunitx(tau_0_val, tau_0_err))
    print('tau_t', siunitx(tau_t_val, tau_t_err))
    print('tau_f', siunitx(tau_f_val, tau_f_err))
    print('tau_bar', siunitx(tau_bar_val, tau_bar_err))

    print('popt', siunitx(popt_val, popt_err))
    print('popt_lin1', siunitx(popt_lin1_val, popt_lin1_err))
    print('popt_lin2', siunitx(popt_lin2_val, popt_lin2_err))

    print(x.shape)
    print(y_lin1_val.shape)

    pl.plot(time, counts, color='black')
    pl.fill_between(x, y_val - y_err, y_val + y_err, alpha=0.5, color='red')
    pl.fill_between(x, y_lin1_val - y_lin1_err, y_lin1_val + y_lin1_err, alpha=0.5, color='blue')
    pl.fill_between(x, y_lin2_val - y_lin2_err, y_lin2_val + y_lin2_err, alpha=0.5, color='blue')
    counts_smooth = scipy.ndimage.filters.gaussian_filter1d(counts, 8)
    pl.plot(time, counts_smooth, color='green')
    pl.plot(x, y_lin1_val, color='blue')
    pl.plot(x, y_lin2_val, color='blue')
    pl.xlabel('Time / ns')
    pl.ylabel('Counts')
    dandify_plot()
    #pl.xlim((8, 20))
    pl.ylim((0.1, np.max(counts)*1.1))
    pl.savefig('_build/mpl-lifetime-acryl.pdf')
    pl.savefig('_build/mpl-lifetime-acryl.png')
    pl.yscale('log')
    pl.savefig('_build/mpl-lifetime-acryl-log.pdf')
    pl.savefig('_build/mpl-lifetime-acryl-log.png')
    #pl.show()
    pl.clf()


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
        for a in range(10):
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
    return slope_val, width_val*slope_val


def get_indium_data(T, slope_val, width):
    files = glob.glob('Data/in-*.txt')

    temps_val = []
    temps_err = []

    all_counts = []

    all_tau_0_dist = []
    all_tau_bar_dist = []
    all_tau_f_dist = []
    all_tau_t_dist = []

    all_intens_0_dist = []
    all_intens_t_dist = []

    all_lifetime_y_dist = []
    all_lifetime_popt_dist = []

    all_sigma_c_dist = []

    # Process lifetime curves with bootstrap.
    for sample_id in range(BOOTSTRAP_SAMPLES):
        print('Bootstrap sample', sample_id, 'running …')

        results = []

        for file_ in sorted(files):
            print('Working on lifetime spectrum', file_)

            if sample_id == 0:
                temp_lower, temp_upper = get_temp(file_)
                temp_mean = (temp_lower + temp_upper)/2
                temp_err = temp_upper - temp_mean
                temps_val.append(temp_mean)
                temps_err.append(temp_err)
                print('Mean temperature:', temp_mean)

            data = np.loadtxt(file_)
            channel = data[:, 0]
            time = slope_val * channel
            counts = data[:, 1]
            boot_counts = bootstrap.redraw_count(counts)

            if sample_id == 0:
                all_counts.append(counts)

            x = np.linspace(np.min(time), np.max(time), 2000)

            sel = (9 < time) & (time < 15)

            fit_func = lambda t, mean, A_0, A_t, tau_0, tau_t, BG: \
                    models.lifetime_spectrum(t, mean, width, A_0, A_t, tau_0, tau_t, BG)
            p0 = [10.5, 210, 190, 0.07, 0.8, 0]
            popt, pconv = op.curve_fit(fit_func, time[sel], boot_counts[sel], p0=p0)
            mean, A_0, A_t, tau_0, tau_t, BG = popt

            intens_0 = A_0 / (A_0 + A_t)
            intens_t = A_t / (A_0 + A_t)
            tau_bar = intens_0 * tau_0 + intens_t * tau_t
            y = fit_func(x, *popt)
            tau_f = 1 / (intens_0 / tau_0 - intens_t / tau_t)
            sigma_c = 1 / tau_0 - 1 / tau_f

            results.append([
                tau_0,
                tau_bar,
                tau_f,
                tau_t,
                intens_0,
                intens_t,
                y,
                popt,
                sigma_c,
            ])


        tau_0_list, tau_bar_list, tau_f_list, tau_t_list, intens_0_list, \
                intens_t_list, lifetime_y_list, lifetime_popt_list, sigma_c_list \
                = zip(*results)

        all_tau_0_dist.append(tau_0_list)
        all_tau_bar_dist.append(tau_bar_list)
        all_tau_f_dist.append(tau_f_list)
        all_tau_t_dist.append(tau_t_list)
        all_intens_0_dist.append(intens_0_list)
        all_intens_t_dist.append(intens_t_list)
        all_lifetime_y_dist.append(lifetime_y_list)
        all_lifetime_popt_dist.append(lifetime_popt_list)
        all_sigma_c_dist.append(sigma_c_list)

    # Generate plots with lifetime curves and fits.
    for temp, counts, lifetime_y_dist in zip(temps_val, all_counts, zip(*all_lifetime_y_dist)):
        print('Creating lifetime plot with temp', temp)
        y_val, y_err = bootstrap.average_and_std_arrays(lifetime_y_dist)

        np.savetxt('_build/xy/lifetime-{}K-data.tsv'.format(int(temp)),
                   bootstrap.pgfplots_error_band(time[0:4000], counts[0:4000], np.sqrt(counts[0:4000])))
        np.savetxt('_build/xy/lifetime-{}K-fit.tsv'.format(int(temp)),
                   np.column_stack([x, y_val]))
        np.savetxt('_build/xy/lifetime-{}K-band.tsv'.format(int(temp)),
                   bootstrap.pgfplots_error_band(x, y_val, y_err))

        if False:
            pl.fill_between(x, y_val - y_err, y_val + y_err, alpha=0.5, color='red')
            pl.plot(time, counts, color='black')
            counts_smooth = scipy.ndimage.filters.gaussian_filter1d(counts, 8)
            pl.plot(time, counts_smooth, color='green')
            pl.plot(x, y_val, color='red')
            pl.xlabel('Time / ns')
            pl.ylabel('Counts')
            dandify_plot()
            pl.xlim((8, 20))
            pl.savefig('_build/mpl-lifetime-{:04d}K.pdf'.format(int(temp)))
            pl.savefig('_build/mpl-lifetime-{:04d}K.png'.format(int(temp)))
            pl.yscale('log')
            pl.savefig('_build/mpl-lifetime-{:04d}K-log.pdf'.format(int(temp)))
            pl.savefig('_build/mpl-lifetime-{:04d}K-log.png'.format(int(temp)))
            pl.clf()

    # Plot the lifetimes.
    taus_0_val, taus_0_err = bootstrap.average_and_std_arrays(all_tau_0_dist)
    taus_t_val, taus_t_err = bootstrap.average_and_std_arrays(all_tau_t_dist)
    taus_f_val, taus_f_err = bootstrap.average_and_std_arrays(all_tau_f_dist)
    taus_bar_val, taus_bar_err = bootstrap.average_and_std_arrays(all_tau_bar_dist)
    pl.errorbar(temps_val, taus_0_val, xerr=temps_err, yerr=taus_0_err,
                label=r'$\tau_0$', linestyle='none', marker='+')
    pl.errorbar(temps_val, taus_bar_val, xerr=temps_err, yerr=taus_bar_err,
                label=r'$\bar\tau$', linestyle='none', marker='+')
    pl.errorbar(temps_val, taus_t_val, xerr=temps_err, yerr=taus_t_err,
                label=r'$\tau_\mathrm{t}$', linestyle='none', marker='+')
    pl.errorbar(temps_val, taus_f_val, xerr=temps_err, yerr=taus_f_err,
                label=r'$\tau_\mathrm{f}$', linestyle='none', marker='+')
    pl.xlabel('T / K')
    pl.ylabel(r'$\tau$ / ns')
    dandify_plot()
    pl.savefig('_build/mpl-tau_0-tau_t.pdf')
    pl.savefig('_build/mpl-tau_0-tau_t.png')
    pl.clf()
    np.savetxt('_build/xy/tau_0.tsv',
               np.column_stack([temps_val, taus_0_val, taus_0_err]))
    np.savetxt('_build/xy/tau_t.tsv',
               np.column_stack([temps_val, taus_t_val, taus_t_err]))
    np.savetxt('_build/xy/tau_f.tsv',
               np.column_stack([temps_val, taus_f_val, taus_f_err]))
    np.savetxt('_build/xy/tau_bar.tsv',
               np.column_stack([temps_val, taus_bar_val, taus_bar_err]))

    # Plot relative intensities.
    all_intens_0_val, all_intens_0_err = bootstrap.average_and_std_arrays(all_intens_0_dist)
    all_intens_t_val, all_intens_t_err = bootstrap.average_and_std_arrays(all_intens_t_dist)
    pl.errorbar(temps_val, all_intens_0_val, xerr=temps_err, yerr=all_intens_0_err,
                label=r'$A_0$', linestyle='none', marker='+')
    pl.errorbar(temps_val, all_intens_t_val, xerr=temps_err, yerr=all_intens_t_err,
                label=r'$A_\mathrm{t}$', linestyle='none', marker='+')
    pl.xlabel('T / K')
    pl.ylabel(r'Relative Intensity')
    dandify_plot()
    pl.savefig('_build/mpl-intensities.pdf')
    pl.savefig('_build/mpl-intensities.png')
    pl.clf()

    inv_temps = 1 / np.array(temps_val)
    results = []
    x = np.linspace(np.min(inv_temps), np.max(inv_temps), 1000)
    for all_sigma_c in all_sigma_c_dist:
        p0 = [11, 240]
        print('inv_temps:', inv_temps)
        print('all_sigma_c:', all_sigma_c)
        popt, pconv = op.curve_fit(exp_decay, inv_temps, all_sigma_c, p0=p0)
        y = exp_decay(x, *popt)

        kelvin_to_eV = 8.621738e-5

        results.append([
            popt,
            popt[1] * kelvin_to_eV,
            y,
        ])

    popt_dist, Ht_eV_dist, arr_y_dist = zip(*results)

    popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)
    print('popt:', siunitx(popt_val, popt_err))
    Ht_eV_val, Ht_eV_err = bootstrap.average_and_std_arrays(Ht_eV_dist)
    arr_y_val, arr_y_err = bootstrap.average_and_std_arrays(arr_y_dist)
    sigma_c_val, sigma_c_err = bootstrap.average_and_std_arrays(all_sigma_c_dist)

    pl.fill_between(x, arr_y_val - arr_y_err, arr_y_val + arr_y_err, alpha=0.5, color='red')
    pl.plot(x, arr_y_val, color='red')
    pl.errorbar(inv_temps, sigma_c_val, yerr=sigma_c_err, marker='+', linestyle='none', color='black')
    pl.xlabel(r'$1 / T$')
    pl.ylabel(r'$\sigma C_t(T)$')
    pl.savefig('_build/mpl-arrhenius.pdf')
    pl.savefig('_build/mpl-arrhenius.png')
    pl.clf()

    np.savetxt('_build/xy/arrhenius-data.tsv',
               np.column_stack([inv_temps, sigma_c_val, sigma_c_err]))
    np.savetxt('_build/xy/arrhenius-fit.tsv',
               np.column_stack([x, arr_y_val]))
    np.savetxt('_build/xy/arrhenius-band.tsv',
               bootstrap.pgfplots_error_band(x, arr_y_val, arr_y_err))

    print('Ht_eV:', siunitx(Ht_eV_val, Ht_eV_err))

    pl.errorbar(temps_val, taus_bar_val, xerr=temps_err, yerr=taus_bar_err,
                label=r'$\bar\tau$', linestyle='none', marker='+')
    dandify_plot()
    pl.xlabel('T / K')
    pl.ylabel(r'$\bar\tau$ / ns')
    pl.savefig('_build/mpl-s_curve.pdf')
    pl.savefig('_build/mpl-s_curve.png')
    pl.clf()
    np.savetxt('_build/xy/s_curve.tsv',
               np.column_stack([temps_val, taus_bar_val, taus_bar_err]))


def s_curve(T, sigma_S, H_t, tau_t, tau_f):
    assert not isinstance(T, list), "x-value input in fit must be np.array."
    sigma_S_exp = sigma_S * np.exp(- H_t/T)
    return tau_f * (1 + sigma_S_exp * tau_t) / (1 + sigma_S_exp * tau_f)


def redraw_count(a):
    '''
    Takes a ``np.array`` with counts and re-draws the counts from the implicit
    Gaussian distribution with width ``sqrt(N)``.
    '''
    out = [random.gauss(x, np.sqrt(x)) for x in a]
    return np.array(out).reshape(a.shape)


def redraw(X_val, X_err):
    val_boot = [random.gauss(val, err) for val, err in zip(X_val, X_err)]
    return np.array(val_boot)


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

    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    options = parser.parse_args()

    prepare_files(T)
    lifetime(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
