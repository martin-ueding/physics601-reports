#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import glob
import json
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


def lifetime_spectrum(t, mean, width, A_0, A_t, tau_0, tau_t, BG):
    return A_0/(2*tau_0) * np.exp((width**2-2*tau_0*(t-mean))/(2*tau_0**2)) \
            * (sp.erf((width**2+tau_0*mean)/(np.sqrt(2)*width*tau_0)) \
            + sp.erf((tau_0*(t-mean)-width**2)/(np.sqrt(2)*width*tau_0))) \
            + A_t/(2*tau_t) * np.exp((width**2-2*tau_t*(t-mean))/(2*tau_t**2)) \
            * (sp.erf((width**2+tau_t*mean)/(np.sqrt(2)*width*tau_t)) \
            + sp.erf((tau_t*(t-mean)-width**2)/(np.sqrt(2)*width*tau_t))) \
            + BG


def linear(x, a, b):
    return a * x + b


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
    lifetime_spectra(T, slope, width)


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
    return slope_val, width_val*slope_val

def get_indium_data(T, slope_val, width):
    files = glob.glob('Data/in-*.txt')

    all_life = []

    temps_val = []
    temps_err = []
    life = []

    tau_ts = []
    tau_0s = []

    for file_ in sorted(files):
        print('Working on lifetime spectrum', file_)
        temp_lower, temp_upper = get_temp(file_)
        temp_mean = (temp_lower + temp_upper)/2
        temp_err = temp_upper - temp_mean
        temps_val.append(temp_mean)
        temps_err.append(temp_err)

        data = np.loadtxt(file_)
        channel = data[:,0]
        time = slope_val * channel
        counts = data[:,1]

        x = np.linspace(np.min(time), np.max(time), 2000)

        results = []
        life_mean = []
        y_dist = []
        for a in range(10):
            boot_counts = redraw_count(counts)
            lifetime_spectrum_fixed_width = lambda t, mean, A_0, A_t, tau_0, tau_t, BG: lifetime_spectrum(t, mean, width, A_0, A_t, tau_0, tau_t, BG)
            popt, pconv = op.curve_fit(lifetime_spectrum_fixed_width, time, boot_counts, p0=[10.5, 210, 190, 0.07, 0.8, 0])
            # popt, pconv = op.curve_fit(lifetime_spectrum, time, boot_counts, p0=[10.5, 0.3, 210, 190, 0.07, 0.8, 0])
            results.append(popt)
            life_mean.append((popt[1]*popt[3] + popt[2]*popt[4]) / (popt[1] + popt[2]))
            y_dist.append(lifetime_spectrum_fixed_width(x, *popt))

        all_life.append(life_mean)

        popt_val, popt_err = bootstrap.average_and_std_arrays(results)
        life_mean_val, life_mean_err = bootstrap.average_and_std_arrays(life_mean)
        life.append(life_mean_val)

        y_val, y_err = bootstrap.average_and_std_arrays(y_dist)

        pl.fill_between(x, y_val - y_err, y_val + y_err, alpha=0.5, color='red')
        pl.plot(time, counts, color='black')
        pl.plot(x, y_val, color='red')
        pl.xlabel('Time / ns')
        pl.ylabel('Counts')
        pl.tight_layout()
        pl.grid(True)
        pl.margins(0.05)
        pl.savefig('_build/mpl-lifetime-{:04d}_dK.pdf'.format(int(temp_mean*10)))
        pl.yscale('log')
        pl.savefig('_build/mpl-lifetime-{:04d}_dK-log.pdf'.format(int(temp_mean*10)))
        pl.clf()

        
        # mean_val, width_val, A_0_val, A_t_val, tau_0_val, tau_t_val, BG_val = popt_val
        # mean_err, width_err, A_0_err, A_t_err, tau_0_err, tau_t_err, BG_err = popt_err
        mean_err, A_0_val, A_t_val, tau_0_val, tau_t_val, BG_val = popt_val
        mean_err, A_0_err, A_t_err, tau_0_err, tau_t_err, BG_err = popt_err

        tau_0s.append(tau_0_val)
        tau_ts.append(tau_t_val)

    return all_life, temps_val, temps_err, life, tau_0s, tau_ts


INDIUM_FILE = '_build/indium.pickle'


def load_indium_data():
    with open(INDIUM_FILE, 'rb') as f:
        return pickle.load(f)


def save_indium_data(all_life, temps_val, temps_err, life, tau_0s, tau_ts):
    with open(INDIUM_FILE, 'wb') as f:
        pickle.dump([all_life, temps_val, temps_err, life, tau_0s, tau_ts], f)


def lifetime_spectra(T, slope_val, width):
    if os.path.isfile(INDIUM_FILE):
        all_life, temps_val, temps_err, life, tau_0s, tau_ts = load_indium_data()
    else:
        all_life, temps_val, temps_err, life, tau_0s, tau_ts = get_indium_data(T, slope_val, width)
        save_indium_data(all_life, temps_val, temps_err, life, tau_0s, tau_ts)

    popt_dist = []
    y_dist =[]
    x = np.linspace(np.min(temps_val), np.max(temps_val), 200)
    life_val, life_err = bootstrap.average_and_std_arrays(np.array(all_life).T)
    

    # From here on >>without<< bootstrap

    temps_val = np.array(temps_val)
    life_val = np.array(life_val)

    tau_t = np.mean(tau_ts)

    def s_curve(T, sigma_S, H_t,
                #tau_t,
                tau_f):
        assert not isinstance(T, list), "x-value input in fit must be np.array."
        sigma_S_exp = sigma_S * np.exp(- H_t/T)
        return tau_f * (1 + sigma_S_exp * tau_t) / (1 + sigma_S_exp * tau_f)

    # p0=[4.2e10, 7.41e3, .352, .330]
    p0=[1.e8,
        5.7e5,
       # .352,
        .330]

    try:
        popt, pconv = op.curve_fit(s_curve, temps_val, life_val, sigma=life_err,
                                  # p0=p0
                                  )
    except RuntimeError as e:
        print(e)
        print('Showing the plot with initial parameters.')
        pl.errorbar(temps_val, life_val, yerr=life_err, linestyle="none", marker="o")
        y = s_curve(x, *p0)
        pl.plot(x, y)
        pl.show()
        pl.clf()
    else:
        print(popt)
        y = s_curve(x, *popt)

    # print(siunitx(popt_val, popt_err))

    print('Showing the plot with actual fit curve.')
    pl.errorbar(temps_val, life_val, xerr=temps_err, yerr=life_err, linestyle="none", marker="+")
    pl.plot(x, y)
    pl.show()
    pl.clf()

    # From here on >>with<< bootstrap

    # for temp_life in zip(*all_life):
    #     temps_boot = redraw(temps_val, temps_err)
    #     leave_out = random.randint(0, len(temps_boot) - 1)
    #     np.delete = lambda x, y: x
    #     temps_fit = np.delete(temps_boot, leave_out)
    #     life_val_fit = np.delete(temp_life, leave_out)
    #     life_err_fit = np.delete(life_err, leave_out)

    #     try:
    #         popt, pconv = op.curve_fit(s_curve, temps_fit, life_val_fit,
    #                                    sigma=life_err_fit, p0=p0)
    #     except RuntimeError as e:
    #         print(e)
    #         pl.errorbar(temps_fit, life_val_fit, yerr=life_err_fit, linestyle="none", marker="o")
    #         y = s_curve(x, *p0)
    #         pl.plot(x, y)
    #         pl.show()
    #         pl.clf()
    #     else:
    #         print(popt)
    #         popt_dist.append(popt)
    #         y = s_curve(x, *popt)
    #         y_dist.append(y)

    # y_val, y_err = bootstrap.average_and_std_arrays(y_dist)
    # popt_val, popt_err = bootstrap.average_and_std_arrays(popt_dist)

    # print(siunitx(popt_val, popt_err))

    # pl.errorbar(temps_val, life_val, xerr=temps_err, yerr=life_err, linestyle="none", marker="+")
    # pl.plot(x, y_val)
    # pl.plot(x, y_val+y_err)
    # pl.plot(x, y_val-y_err)
    # pl.show()
    # pl.clf()



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
