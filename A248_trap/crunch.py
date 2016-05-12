#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import itertools
import json
import os
import pprint
import random
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
import trek


def linear(x, a, b):
    return a * x + b


def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 


def loading(x, a, b, offset):
    return a*(1-np.exp(-b*(x-offset)))


def errorfunction(x, power, diam, x_offs):
    return power / 2 * sp.erfc(np.sqrt(8) / diam * (x - x_offs))


def cos_squared(x, ampl, x_offs, y_offs):
    return ampl * (np.cos(x - x_offs))**2 + y_offs


def subtract_images(number_str):
    img_with = scipy.misc.imread('Figures/{}-mit.bmp'.format(number_str))
    img_without = scipy.misc.imread('Figures/{}-ohne.bmp'.format(number_str))

    difference = np.subtract(img_with.astype(int), img_without.astype(int))

    print(img_with.dtype)

    print(np.min(img_with), np.max(img_with))
    print(np.min(img_without), np.max(img_without))
    print(np.min(difference), np.max(difference))

    old_min = np.min(difference)
    old_max = np.max(difference)
    span = old_max - old_min

    #difference = (difference - old_min) * 255 / span

    print(np.min(difference), np.max(difference))
    print()

    return difference


def add_images(image_1, image_2):
    img_1 = scipy.misc.imread(image_1)
    img_2 = scipy.misc.imread(image_2)

    sum_ = np.add(img_1.astype(int), img_2.astype(int))

    old_min = np.min(sum_)
    old_max = np.max(sum_)
    span = old_max - old_min

    #difference = (difference - old_min) * 255 / span

    print(np.min(sum_), np.max(sum_))
    print()

    return sum_


def invert_image(image):
    return 255 - image


def dandify_plot():
    pl.margins(0.05)
    pl.tight_layout()
    pl.grid(True)
    pl.legend(loc='best')


def job_doppler_free(T):
    osci08_x1, osci08_y1, osci08_x2, osci08_y2 = trek.load_dir('0008') 
    osci20_x1, osci20_y1, osci20_x2, osci20_y2 = trek.load_dir('0020')

    np.savetxt('_build/xy/doppler-free-pumping.tsv', np.column_stack([osci08_x1, osci08_y1]))
    np.savetxt('_build/xy/doppler-free-cooling.tsv', np.column_stack([osci20_x1, osci20_y1]))


def fit_osci_peak(x, y, xmin, xmax, basename=None):
    sel = (xmin < x) & (x < xmax)
    popt, pconv = op.curve_fit(gauss, x[sel], y[sel])
    perr = np.sqrt(pconv.diagonal())
    
    if basename is not None:
        interp_x = np.linspace(xmin, xmax, 100)
        interp_y = gauss(interp_x, *popt)
        np.savetxt('_build/xy/'+basename, np.column_stack([interp_x, interp_y]))

    return popt[0], perr[0]


def job_scan_cooling(T):
    osci19_x1, osci19_y1, osci19_x2, osci19_y2 = trek.load_dir('0019')
    osci20_x1, osci20_y1, osci20_x2, osci20_y2 = trek.load_dir('0020')

    np.savetxt('_build/xy/scan-cooling-mot-input.tsv', np.column_stack([osci20_x1, osci20_y1]))
    np.savetxt('_build/xy/scan-cooling-mot-output.tsv', np.column_stack([osci20_x2, osci20_y2]))
    np.savetxt('_build/xy/scan-cooling-no_mot-input.tsv', np.column_stack([osci19_x1, osci19_y1]))
    np.savetxt('_build/xy/scan-cooling-no_mot-output.tsv', np.column_stack([osci19_x2, osci19_y2]))
    np.savetxt('_build/xy/scan-cooling-difference-output.tsv', np.column_stack([osci19_x2, osci19_y2 - osci20_y2]))

    peaks_val = []
    peaks_err = []

    peak_val, peak_err = fit_osci_peak(osci20_x1, osci20_y1, -1.00, -0.65, 'scan-cooling-mot-input-fit1.tsv')
    peaks_val.append(peak_val)
    peaks_err.append(peak_err)
    peak_val, peak_err = fit_osci_peak(osci20_x1, osci20_y1, -0.40, -0.10, 'scan-cooling-mot-input-fit2.tsv')
    peaks_val.append(peak_val)
    peaks_err.append(peak_err)
    peak_val, peak_err = fit_osci_peak(osci20_x1, osci20_y1, 0.80, 1.07, 'scan-cooling-mot-input-fit3.tsv')
    peaks_val.append(peak_val)
    peaks_err.append(peak_err)

    spacings = np.array([0, 31.7, 31.7 + 60.3])
    spacings -= spacings[2]
    popt, pconv = op.curve_fit(linear, spacings, peaks_val, sigma=peaks_err)

    np.savetxt('_build/xy/scan-cooling-spacing-data.tsv',
               np.column_stack([spacings, peaks_val, peaks_err]))

    fit_x = np.linspace(min(spacings), max(spacings), 10)
    fit_y = linear(fit_x, *popt)

    np.savetxt('_build/xy/scan-cooling-spacing-data.tsv',
               np.column_stack([spacings, peaks_val, peaks_err]))
    np.savetxt('_build/xy/scan-cooling-spacing-fit.tsv',
               np.column_stack([fit_x, fit_y]))

    detuning_x = (osci19_x2 - popt[1]) / popt[0]
    detuning_y = osci19_y2 - osci20_y2

    np.savetxt('_build/xy/scan-cooling-detuning.tsv',
               np.column_stack([detuning_x, detuning_y]))

    sel = (-18 < detuning_x) & (detuning_x < -9)
    p0 = [-15, 3, 1]
    popt, pconv = op.curve_fit(gauss, detuning_x[sel], detuning_y[sel], p0=p0)
    fit_x = np.linspace(-18, -9, 100)
    fit_y = gauss(fit_x, *popt)
    np.savetxt('_build/xy/scan-cooling-detuning-fit.tsv',
               np.column_stack([fit_x, fit_y]))

    perr = np.sqrt(pconv.diagonal())

    T['detuning'] = siunitx(popt[0], perr[0])

    return popt[0], perr[0]


def job_scan_pumping(T):
    osci17_x1, osci17_y1, osci17_x2, osci17_y2 = trek.load_dir('0017')
    osci18_x1, osci18_y1, osci18_x2, osci18_y2 = trek.load_dir('0018')

    np.savetxt('_build/xy/scan-pumping-mot-input.tsv', np.column_stack([osci18_x1, osci18_y1]))
    np.savetxt('_build/xy/scan-pumping-mot-output.tsv', np.column_stack([osci18_x2, osci18_y2]))
    np.savetxt('_build/xy/scan-pumping-no_mot-input.tsv', np.column_stack([osci17_x1, osci17_y1]))
    np.savetxt('_build/xy/scan-pumping-no_mot-output.tsv', np.column_stack([osci17_x2, osci17_y2]))
    np.savetxt('_build/xy/scan-pumping-difference-output.tsv', np.column_stack([osci17_x2, osci17_y2 - osci18_y2]))


def job_loading(T):
    res_max = []
    res_slope = []
    for i, directory in zip(itertools.count(), ['0002', '0003', '0004', '0005', '0006', '0007']):
        data_x, data_y = trek.load_dir(directory)
        data_x, data_y = data_x[120:-500], data_y[120:-500]

        lower = np.where(data_y > .095)[0][0]

        fit_x = np.linspace(data_x[lower-20], data_x[-1], 100)
        popt, pconv = op.curve_fit(loading, data_x[lower:], data_y[lower:])
        res_max.append(popt[0])
        res_slope.append(popt[1])
        fit_y = loading(fit_x, *popt)
        # pl.plot(data_x, data_y)
        # pl.plot(fit_x, fit_y)
        # pl.show()
        # pl.clf()

        np.savetxt('_build/xy/loading-{}-data.tsv'.format(i),
                   np.column_stack([data_x, data_y * 100]))
        np.savetxt('_build/xy/loading-{}-fit.tsv'.format(i),
                   np.column_stack([fit_x, fit_y * 100]))

    maximum_val, maximum_err = np.mean(res_max), np.std(res_max)
    slope_val, slope_err = np.mean(res_slope), np.std(res_slope)


def get_mot_power_nw(T):
    data = np.loadtxt('Data/mot-intensity.tsv')
    mot_with = data[:,0]
    und = data[:,1]
    # err = data[:,2]
    mot_w_o = mot_with - und
    
    power_mean = np.mean(mot_w_o)
    power_err = np.std(mot_w_o)

    T['power_mot'] = siunitx(power_mean, power_err)
    T['power_mot_table'] = list(zip(
        siunitx(mot_with),
        siunitx(und),
        siunitx(mot_w_o)
    ))

    return power_mean, power_err


def job_diameter(T):
    data = np.loadtxt('Data/diameter.tsv')
    position = data[:,0]
    power = data[:,1]

    x = np.linspace(np.min(position), np.max(position), 100)
    popt, pconv = op.curve_fit(errorfunction, position, power, p0=[3.4, .4, 29.5])
    perr = np.sqrt(pconv.diagonal())

    y = errorfunction(x, *popt)
    # pl.plot(position, power, linestyle="none", marker="+")
    # pl.plot(x, y)
    # pl.show()
    # pl.clf()

    np.savetxt('_build/xy/diameter-data.tsv',
               np.column_stack([position, power]))
    np.savetxt('_build/xy/diameter-fit.tsv',
               np.column_stack([x, y]))

    T['beam_diameter_table'] = list(zip(
        siunitx(position),
        siunitx(power)
    ))

    T['beam_diameter'] = siunitx(popt[1], perr[1])
    T['beam_power'] = siunitx(popt[0], perr[0])

    return popt[1], perr[1]


def job_lambda_4(T):
    for name in ['front', 'behind']:
        data = np.loadtxt('Data/lambda-{}.tsv'.format(name))
        angle = data[:,0]
        power = data[:,1]

        if name=='front':
            for i in range(len(angle)):
                if angle[i] > 100:
                    angle[i] = angle[i] - 360

        angle *= (np.pi / 180)
        x = np.linspace(np.min(angle), np.max(angle), 100)
        popt, pconv = op.curve_fit(cos_squared, angle, power, p0=[400,1, 350])
        y = cos_squared(x, *popt)
        print(*popt)

        pl.plot(angle, power, linestyle="none", marker="+")
        pl.plot(x, y)
        #pl.show()
        pl.clf()

        np.savetxt('_build/xy/lambda_{}.tsv'.format(name), np.column_stack([angle, power]))
        np.savetxt('_build/xy/lambda_{}_fit.tsv'.format(name), np.column_stack([x, y]))


def get_mm_per_pixel():
    # Those points are 1 cm apart on the image
    v1 = np.array([460, 186])
    v2 = np.array([720, 182])
    length_px = np.linalg.norm(v1 - v2)
    return 10 / length_px


def job_mot_size(T):
    diff3 = subtract_images('03')
    scipy.misc.imsave('_build/difference-3.png', diff3)
    scipy.misc.imsave('_build/difference-3-inv.png', invert_image(diff3))
    diff4 = subtract_images('04')
    scipy.misc.imsave('_build/difference-4.png', diff4)

    superposition = add_images('Figures/scale.bmp','_build/difference-3-inv.png')
    scipy.misc.imsave('_build/motsize.png', superposition[147:147+308, 402:402+329])

    mm_per_pixel = get_mm_per_pixel()

    selection = diff3[221:221+167, 513:513+150]

    scipy.misc.imsave('_build/mot-crop.png', selection)

    s = selection.shape
    total_pixels = s[0] * s[1]

    white_pixels = np.sum(selection - np.min(selection)) / (np.max(selection) - np.min(selection))

    radius_px = np.sqrt(white_pixels / np.pi)
    radius_mm = radius_px * mm_per_pixel
    volume_mm3 = 4/3 * np.pi * radius_mm**3

    T['mot_mm_per_pixel'] = siunitx(mm_per_pixel)
    T['mot_total_pixels'] = siunitx(total_pixels)
    T['mot_white_pixels'] = siunitx(white_pixels)
    T['mot_radius_px'] = siunitx(radius_px)
    T['mot_radius_mm'] = siunitx(radius_mm)
    T['mot_volume_mm3'] = siunitx(volume_mm3)


def get_scattering_rate_MHz(T, intens_mw_cm2, detuning_mhz):
    intens_sat_mw_cm2 = 4.1
    natural_width_mhz = 6

    intens_ratio = intens_mw_cm2 / intens_sat_mw_cm2
    detuning_ratio = detuning_mhz / natural_width_mhz

    T['intens_ratio'] = siunitx(intens_ratio)
    T['detuning_ratio'] = siunitx(detuning_ratio)

    return intens_ratio * np.pi * detuning_mhz / (1 + intens_ratio + 4 * detuning_ratio**2)


def job_magnetic_field(T):
    data = np.loadtxt('Data/magnetic.tsv')
    current = data[:, 0]
    intensity_nw = data[:, 1]

    np.savetxt('_build/xy/magnetic.tsv',
               np.column_stack([current, intensity_nw]))


def job_atom_number(T):
    wavelength_nm = 780
    beam_power_mW = 2 * (3.57 + 3.36 + 4.45)
    mot_power_nW_val, mot_power_nW_err = get_mot_power_nw(T)
    lens_distance_cm = 10
    lens_radius_cm = 2.54 / 2
    omega_val = np.pi * lens_radius_cm**2 / (lens_distance_cm**2)
    mot_power_tot_nW_val = mot_power_nW_val * 4 * np.pi / omega_val
    diameter_cm_val, diameter_cm_err = job_diameter(T)
    beam_area_cm_val = np.pi * diameter_cm_val**2 / 4
    intens_mW_cm2_val = beam_power_mW / beam_area_cm_val
    hbar = 1.054571800e-34
    hbar_omega_nW_MHz = hbar * 1e15 * 2 * np.pi * 3e8 / (wavelength_nm * 1e-9)
    detuning_MHz_val, detuning_MHz_err = map(abs, job_scan_cooling(T))
    scattering_rate_MHz = get_scattering_rate_MHz(T, intens_mW_cm2_val, detuning_MHz_val)
    atom_number_val = mot_power_tot_nW_val / hbar_omega_nW_MHz / scattering_rate_MHz

    T['beam_area_cm'] = siunitx(beam_area_cm_val)
    T['total_beam_power_mW'] = siunitx(beam_power_mW)
    T['lens_distance_cm'] = siunitx(lens_distance_cm)
    T['lens_radius_cm'] = siunitx(lens_radius_cm)
    T['mot_omega'] = siunitx(omega_val)
    T['mot_power_nW'] = siunitx(mot_power_nW_val)
    T['mot_power_tot_nW'] = siunitx(mot_power_tot_nW_val)
    T['intens_mW_cm2'] = siunitx(intens_mW_cm2_val)
    T['hbar_omega_nW_MHz'] = siunitx(hbar_omega_nW_MHz)
    T['scattering_rate_MHz'] = siunitx(scattering_rate_MHz)
    T['atom_number'] = siunitx(atom_number_val)
    T['wavelength_nm'] = siunitx(wavelength_nm)


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

    # We use bootstrap and obtain different results every single time. This is
    # bad, therefore we fix the seed here.
    random.seed(0)

    job_atom_number(T)
    job_magnetic_field(T)
    job_mot_size(T)
    job_doppler_free(T)
    job_scan_pumping(T)

    job_loading(T)
    job_lambda_4(T)

    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

    pp = pprint.PrettyPrinter()
    pp.pprint(T)


if __name__ == "__main__":
    main()
