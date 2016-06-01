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

SAMPLES = 100


def linear(x, a, b):
    return a * x + b


def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 


def sinc(x, center, width, amplitude, offset):
    return amplitude * np.sinc((x - center) / width) + offset


def errorfunction(x, power, diam, x_offs):
    return power / 2 * sp.erfc(np.sqrt(8) / diam * (x - x_offs))


def cos_squared(x, ampl, x_offs, y_offs):
    return ampl * (np.cos(2*np.radians(x - x_offs)))**2 + y_offs


def cos_quartic(x, ampl, x_offs, y_offs):
    return ampl * (np.cos(2*np.radians(x - x_offs)))**4 + y_offs


def job_power(T):
    data = np.loadtxt('Data/diode.tsv')
    norm_current = data[:, 0] * 1e-3
    norm_power_val = data[:, 1] * 1e-3
    norm_power_err = np.ones(norm_power_val.shape) * 1e-6
    norm_power_dist = bootstrap.make_dist(norm_power_val, norm_power_err)

    data = np.loadtxt('Data/diode_damped.tsv')
    damp_current = data[:, 0] * 1e-3
    damp_power_val = data[:, 1] * 1e-3
    damp_power_err = data[:, 2] * 1e-3
    damp_power_dist = bootstrap.make_dist(damp_power_val, damp_power_err)

    np.savetxt('_build/xy/diode_normal-data.tsv',
               np.column_stack([norm_current, norm_power_val, norm_power_err]))
    np.savetxt('_build/xy/diode_damped-data.tsv',
               np.column_stack([damp_current, damp_power_val, damp_power_err]))

    hbar_omega = 6.626e-34 * 3e8 / 987e-9
    electron_charge = 1.609e-19

    # Find the threshold current.
    sel = norm_power_val > 1e-3
    slope_dist = []
    quantum_efficiency_dist = []
    threshold_dist = []
    threshold_fit_x = np.linspace(0.05, 0.09, 100)
    threshold_fit_y_dist = []
    # Jackknife fit to find root.
    for i in range(len(norm_power_val[sel])):
        x = np.delete(norm_current[sel], i)
        y_val = np.delete(norm_power_val[sel], i)
        y_err = np.delete(norm_power_err[sel], i)
        popt, pconv = op.curve_fit(linear, x, y_val, sigma=y_err)
        a, b = popt
        root = -b / a
        threshold_dist.append(root)
        threshold_fit_y_dist.append(linear(threshold_fit_x, *popt))
        slope_dist.append(a)
        quantum_efficiency_dist.append(a * electron_charge / hbar_omega)
    threshold_val, threshold_err = bootstrap.average_and_std_arrays(threshold_dist)
    threshold_fit_y_val, threshold_fit_y_err = bootstrap.average_and_std_arrays(threshold_fit_y_dist)
    differential_efficiency_val, differential_efficiency_err = bootstrap.average_and_std_arrays(slope_dist)
    quantum_efficiency_val, quantum_efficiency_err = bootstrap.average_and_std_arrays(quantum_efficiency_dist)

    T['threshold'] = siunitx(threshold_val, threshold_err)
    T['differential_efficiency'] = siunitx(differential_efficiency_val, differential_efficiency_err)
    T['quantum_efficiency'] = siunitx(quantum_efficiency_val, quantum_efficiency_err)

    np.savetxt('_build/xy/diode_normal-band.tsv',
               bootstrap.pgfplots_error_band(threshold_fit_x, threshold_fit_y_val, threshold_fit_y_err))

    # Compare ratios of damped and normal power in the overlap range.
    ratio_dist = []
    x = np.linspace(70.1e-3, 86.9e-3, 20)
    for norm_power, damp_power in zip(norm_power_dist, damp_power_dist):
        norm_inter = scipy.interpolate.interp1d(norm_current, norm_power)
        damp_inter = scipy.interpolate.interp1d(damp_current, damp_power)
        a = norm_inter(x)
        b = damp_inter(x)
        ratio = a / b
        ratio_dist.append(ratio)

    ratio_val, ratio_err = bootstrap.average_and_std_arrays(ratio_dist)

    extinction_dist = np.array(ratio_dist).flatten()
    extinction_val, extinction_err = np.mean(ratio_dist), np.std(ratio_dist)
    T['extinction'] = siunitx(extinction_val, extinction_err)

    np.savetxt('_build/xy/diode-ratio-line.tsv',
               np.column_stack([x, ratio_val]))
    np.savetxt('_build/xy/diode-ratio-band.tsv',
               bootstrap.pgfplots_error_band(x, ratio_val, ratio_err))

    return extinction_dist


def get_optimal_focal_length(beam_radius, refractive_index, wavelength, length):
    optimal_normalized_length = 2.84
    bracket = 2 * beam_radius**2 * refractive_index * np.pi * optimal_normalized_length / (length * wavelength) - 1
    factor = (length / (2 * optimal_normalized_length))**2
    return np.sqrt(bracket * factor)


def job_rayleigh_length(T):
    beam_diameter_val = 3.5e-3
    beam_diameter_err = 0.5e-3
    refractive_index = 2.2
    wavelength = 987e-9
    length = 5e-3
    distance = 60e-3

    beam_radius_val = beam_diameter_val / 2
    beam_radius_err = beam_diameter_err / 2

    T['beam_radius'] = siunitx(beam_radius_val, beam_radius_err)

    beam_radius_dist = bootstrap.make_dist(beam_radius_val, beam_diameter_err)

    theta_dist = [
        np.arctan(beam_radius / distance)
        for beam_radius in beam_radius_dist
    ]
    theta_val, theta_err = bootstrap.average_and_std_arrays(theta_dist)
    T['theta'] = siunitx(theta_val, theta_err)

    waist_dist = [
        wavelength / (np.pi * theta)
        for theta in theta_dist
    ]
    waist_val, waist_err = bootstrap.average_and_std_arrays(waist_dist)
    T['waist_mum'] = siunitx(waist_val / 1e-6, waist_err / 1e-6)

    rayleigh_length_dist = list(itertools.filterfalse(np.isnan, [
        refractive_index * np.pi * waist**2 / wavelength
        for waist in waist_dist
    ]))
    rayleigh_length_val, rayleigh_length_err = bootstrap.average_and_std_arrays(rayleigh_length_dist)
    T['rayleigh_length_mm'] = siunitx(rayleigh_length_val / 1e-3, rayleigh_length_err / 1e-3, error_digits=2)

    normalized_length_dist = list([
        length / (2 * rayleigh_length)
        for rayleigh_length in rayleigh_length_dist
    ])
    normalized_length_val, normalized_length_err = bootstrap.average_and_std_arrays(normalized_length_dist)
    T['normalized_length'] = siunitx(normalized_length_val, normalized_length_err, error_digits=2)

    t = (normalized_length_val - 2.84) / normalized_length_err
    T['boyd_kleinman_ttest_t'] = siunitx(t)

    optimal_focal_length_dist = list([
        get_optimal_focal_length(beam_radius, refractive_index, wavelength, length)
        for beam_radius in beam_radius_dist
    ])
    optimal_focal_length_val, optimal_focal_length_err = bootstrap.average_and_std_arrays(optimal_focal_length_dist)
    T['optimal_focal_length_mm'] = siunitx(optimal_focal_length_val / 1e-3, optimal_focal_length_err / 1e-3, error_digits=2)


def make_lissajous(angle, ratio, offset, filename):
    x = np.sin(angle)
    y = np.sin(angle * ratio + offset)
    np.savetxt(filename, np.column_stack([x, y]))


def job_lissajous(T):
    angle = np.linspace(0, 8 * np.pi, 1000)

    make_lissajous(angle, 2, 0, '_build/xy/lissajous_2_0.tsv')
    make_lissajous(angle, 2, 0.2, '_build/xy/lissajous_2_02.tsv')
    make_lissajous(angle, 2, 1, '_build/xy/lissajous_2_1.tsv')

    make_lissajous(angle, 2.1, 0, '_build/xy/lissajous_21_0.tsv')
    make_lissajous(angle, 2.1, 0.7, '_build/xy/lissajous_21_07.tsv')
    make_lissajous(angle, 2.3, 2.4, '_build/xy/lissajous_23_24.tsv')

    make_lissajous(angle, 1, 0, '_build/xy/lissajous_1_0.tsv')
    make_lissajous(angle, 1, 1, '_build/xy/lissajous_1_1.tsv')
    make_lissajous(angle, 3, 0, '_build/xy/lissajous_3_0.tsv')


def job_variable_attenuator(T, extinction_dist):
    data = np.loadtxt('Data/variable.tsv')
    angle = data[:, 0]
    power_val = data[:, 1] * 1e-6
    power_err = np.ones(power_val.shape) * 1e-6

    power_dist = bootstrap.make_dist(power_val, power_err, n=len(extinction_dist))

    fit_x = np.linspace(np.min(angle), np.max(angle), 200)
    fit_y_dist = []
    angle_offset_dist = []
    a_dist = []
    b_dist = []
    popt_dist = []
    extinction_ratio_dist = []
    for power in power_dist:
        popt, pconv = op.curve_fit(cos_squared, angle, power, p0=[1.5, 0, 0])
        fit_y_dist.append(cos_squared(fit_x, *popt))
        angle_offset_dist.append(popt[1])
        a = popt[0]
        b = popt[2]
        a_dist.append(a)
        b_dist.append(b)
        popt_dist.append(popt)
        extinction_ratio_dist.append((a + b) / b)
    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(fit_y_dist)
    angle_offset_val, angle_offset_err = bootstrap.average_and_std_arrays(angle_offset_dist)
    a_val, a_err = bootstrap.average_and_std_arrays(a_dist)
    b_val, b_err = bootstrap.average_and_std_arrays(b_dist)
    extinction_ratio_val, extinction_ratio_err = bootstrap.average_and_std_arrays(extinction_ratio_dist)

    np.savetxt('_build/xy/variable-data.tsv',
               np.column_stack([angle, power_val, power_err]))
    np.savetxt('_build/xy/variable-fit.tsv',
               np.column_stack([fit_x, fit_y_val]))
    np.savetxt('_build/xy/variable-band.tsv',
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))

    T['variable_angle_offset'] = siunitx(angle_offset_val, angle_offset_err)
    T['variable_a'] = siunitx(a_val, a_err)
    T['variable_b'] = siunitx(b_val, b_err)
    T['extinction_ratio'] = siunitx(extinction_ratio_val, extinction_ratio_err)

    return popt_dist
    

def job_temperature_dependence(T):
    data = np.loadtxt('Data/temperature.tsv')
    temp = data[:, 0]
    power_val = data[:, 1] * 1e-6
    power_err = np.ones(power_val.shape) * 1e-6
    power_dist = bootstrap.make_dist(power_val, power_err)

    p0 = [36.5, 1, 36-6, 2e-6]
    fit_x = np.linspace(np.min(temp), np.max(temp), 300)
    popt_dist = []
    fit_y_dist = []
    for power in power_dist:
        popt, pconv = op.curve_fit(sinc, temp, power, p0=p0)
        fit_y_dist.append(sinc(fit_x, *popt))
        popt_dist.append(popt)

    center_dist, width_dist, amplitude_dist, offset_dist = zip(*popt_dist)

    center_val, center_err = bootstrap.average_and_std_arrays(center_dist)
    width_val, width_err = bootstrap.average_and_std_arrays(width_dist)
    amplitude_val, amplitude_err = bootstrap.average_and_std_arrays(amplitude_dist)
    offset_val, offset_err = bootstrap.average_and_std_arrays(offset_dist)

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(fit_y_dist)

    np.savetxt('_build/xy/temperature-data.tsv',
               np.column_stack([temp, power_val, power_err]))
    np.savetxt('_build/xy/temperature-fit.tsv',
               np.column_stack([fit_x, fit_y_val]))
    np.savetxt('_build/xy/temperature-band.tsv',
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))

    T['temp_center'] = siunitx(center_val, center_err)
    T['temp_width'] = siunitx(width_val, width_err)
    T['temp_amplitude'] = siunitx(amplitude_val, amplitude_err)
    T['temp_offset'] = siunitx(offset_val, offset_err)


def job_harmonic_power(T, extinction_dist, input_popt_dist):
    data = np.loadtxt('Data/harmonic_splitter.tsv')
    angle = data[:, 0]
    power_val = data[:, 1] * 1e-6
    power_err = data[:, 2] * 1e-6

    power_dist = bootstrap.make_dist(power_val, power_err)

    fit_x = np.linspace(np.min(angle), np.max(angle), 200)
    fit_y_dist = []
    angle_offset_dist = []
    a_dist = []
    b_dist = []
    popt_dist = []
    for power in power_dist:
        popt, pconv = op.curve_fit(cos_quartic, angle, power, p0=[1.5e-5, 0, 0])
        fit_y_dist.append(cos_quartic(fit_x, *popt))
        angle_offset_dist.append(popt[1])
        a = popt[0]
        b = popt[2]
        a_dist.append(a)
        b_dist.append(b)
        popt_dist.append(popt)
    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(fit_y_dist)
    angle_offset_val, angle_offset_err = bootstrap.average_and_std_arrays(angle_offset_dist)
    a_val, a_err = bootstrap.average_and_std_arrays(a_dist)
    b_val, b_err = bootstrap.average_and_std_arrays(b_dist)

    np.savetxt('_build/xy/harmonic-splitter-data.tsv',
               np.column_stack([angle, power_val, power_err]))
    np.savetxt('_build/xy/harmonic-splitter-fit.tsv',
               np.column_stack([fit_x, fit_y_val]))
    np.savetxt('_build/xy/harmonic-splitter-band.tsv',
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    T['splitter_angle_offset'] = siunitx(angle_offset_val, angle_offset_err)
    T['splitter_a'] = siunitx(a_val, a_err)
    T['splitter_b'] = siunitx(b_val, b_err)

    efficiency_dist = []
    efficiency_sq_dist = []
    for extinction, input_popt, popt in zip(extinction_dist, input_popt_dist, popt_dist):
        efficiency = popt[0] / (input_popt[0] * extinction)
        efficiency_dist.append(efficiency)
        efficiency_sq = popt[0] / (input_popt[0] * extinction)**2
        efficiency_sq_dist.append(efficiency_sq)
    efficiency_val, efficiency_err = bootstrap.average_and_std_arrays(efficiency_dist)
    efficiency_sq_val, efficiency_sq_err = bootstrap.average_and_std_arrays(efficiency_sq_dist)
    T['efficiency'] = siunitx(efficiency_val, efficiency_err)
    T['efficiency_sq'] = siunitx(efficiency_sq_val, efficiency_sq_err)


def job_grating_resolution(T):
    lines_per_m = 600e3
    diameter = 3.5e-3 / 2
    illuminated = diameter * lines_per_m
    relative_error = 1 / illuminated

    T['illuminated'] = siunitx(illuminated)
    T['relative_error'] = siunitx(relative_error)


def job_input_polarization(T):
    data = np.loadtxt('Data/harmonic_bare.tsv')
    angle = data[:, 0]
    power_val = data[:, 1] * 1e-6
    power_err = data[:, 2] * 1e-6
    power_dist = bootstrap.make_dist(power_val, power_err)

    fit_x = np.linspace(np.min(angle), np.max(angle), 200)
    fit_y_dist = []
    angle_offset_dist = []
    a_dist = []
    b_dist = []
    popt_dist = []
    for power in power_dist:
        popt, pconv = op.curve_fit(cos_quartic, angle, power, p0=[1.5e-5, 0, 0])
        fit_y_dist.append(cos_quartic(fit_x, *popt))
        angle_offset_dist.append(popt[1])
        a = popt[0]
        b = popt[2]
        a_dist.append(a)
        b_dist.append(b)
        popt_dist.append(popt)
    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(fit_y_dist)
    angle_offset_val, angle_offset_err = bootstrap.average_and_std_arrays(angle_offset_dist)
    a_val, a_err = bootstrap.average_and_std_arrays(a_dist)
    b_val, b_err = bootstrap.average_and_std_arrays(b_dist)

    np.savetxt('_build/xy/harmonic-bare-data.tsv',
               np.column_stack([angle, power_val, power_err]))
    np.savetxt('_build/xy/harmonic-bare-fit.tsv',
               np.column_stack([fit_x, fit_y_val]))
    np.savetxt('_build/xy/harmonic-bare-band.tsv',
               bootstrap.pgfplots_error_band(fit_x, fit_y_val, fit_y_err))
    T['bare_angle_offset'] = siunitx(angle_offset_val, angle_offset_err)
    T['bare_a'] = siunitx(a_val, a_err)
    T['bare_b'] = siunitx(b_val, b_err)

def michelson_resolution(T):

    d_cm = np.sort(np.loadtxt('Data/lissajous_X.tsv'))
    order = np.arange(0,len(d_cm))

    wavelength = 987e-9

    # get theoretical values
    n_ground_theor = 1 + 1e-8 * (8342.13 + 2406030/(130-1/(wavelength*1e6)**2) + 15997/(38.9-1/(wavelength*1e6)**2))
    T['n_ground_theor'] = n_ground_theor

    n_harm_theor = 1 + 1e-8 * (8342.13 + 2406030/(130-1/(wavelength*.5e6)**2) + 15997/(38.9-1/(wavelength*.5e6)**2)) 
    T['n_harm_theor'] = n_harm_theor

    delta_n_theor = n_harm_theor - n_ground_theor
    T['delta_n_theor'] = delta_n_theor

    # prepare arrays
    slope_dist_cm = []
    offset_dist_cm = []
    delta_n_dist = []
    dev_ratio_dist = []
    res_michelson_dist = []

    fit_y_dist = []

    # perform Jackknife
    for i in range(len(d_cm)):
        x = np.delete(order, i)
        y = np.delete(d_cm, i)

        popt, pconv = op.curve_fit(linear, x, y)

        slope_dist_cm.append(popt[0])
        offset_dist_cm.append(popt[1])

        fit_y_dist.append(linear(x, *popt))

        # experimental value for difference in refractive index
        delta_n = wavelength/(8*popt[0]*1e-2)
        delta_n_dist.append(delta_n)

        # deviation from optimal 1/2 ratio
        dev_ratio_dist.append(.5*delta_n/n_ground_theor)

        # resolution of michelson interferometer
        delta_wavelength = wavelength * (delta_n/n_ground_theor**2)
        res_michelson_dist.append(wavelength/delta_wavelength)

    # get averages and std
    slope_val_cm, slope_err_cm = bootstrap.average_and_std_arrays(slope_dist_cm)
    offset_val_cm, offset_err_cm = bootstrap.average_and_std_arrays(offset_dist_cm)

    fit_y_val, fit_y_err = bootstrap.average_and_std_arrays(fit_y_dist)

    delta_n_val, delta_n_err = bootstrap.average_and_std_arrays(delta_n_dist)

    dev_ratio_val, dev_ratio_err = bootstrap.average_and_std_arrays(dev_ratio_dist)
    dev_ratio_theor = .5*(delta_n_theor/n_ground_theor)

    res_michelson_val, res_michelson_err = bootstrap.average_and_std_arrays(res_michelson_dist)

    # write into T
    T['distance_lissajous_X'] = siunitx(slope_val_cm, slope_err_cm)
    T['delta_n'] = siunitx(delta_n_val, delta_n_err)
    T['dev_ratio'] = siunitx(dev_ratio_val, dev_ratio_err)
    T['dev_ratio_theor'] = siunitx(dev_ratio_theor)
    T['res_michelson'] = siunitx(res_michelson_val, res_michelson_err)

    # write data for plot

    np.savetxt('_build/xy/michelson-band.tsv', bootstrap.pgfplots_error_band(x, fit_y_val, fit_y_err))

    print(siunitx(dev_ratio_val, dev_ratio_err))
    print(siunitx(dev_ratio_theor))
    print(siunitx(res_michelson_val, res_michelson_err))
    

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

    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    job_grating_resolution(T)
    job_input_polarization(T)
    job_temperature_dependence(T)
    extinction_dist = job_power(T)
    input_popt_dist = job_variable_attenuator(T, extinction_dist)
    job_harmonic_power(T, extinction_dist, input_popt_dist)
    job_lissajous(T)
    job_rayleigh_length(T)
    michelson_resolution(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

    pp = pprint.PrettyPrinter()
    print()
    print('Content in T dict:')
    pp.pprint(T)


if __name__ == "__main__":
    main()
