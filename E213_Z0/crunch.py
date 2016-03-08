#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import json
import sys

import matplotlib.pyplot as pl
import numpy as np
import scipy.interpolate
import scipy.misc
import scipy.ndimage.filters
import scipy.optimize as op
import scipy.stats

from unitprint2 import siunitx
import bootstrap

fermi_coupling = 1.6637e-11 # MeV^{-2}
mass_z = 91182 # MeV
sin_sq_weak_mixing = 0.2312
weak_mixing_angle = np.arcsin(np.sqrt(sin_sq_weak_mixing))

def job_decay_widths(T):
    # T_3, Q, N_color
    quantum_numbers = {
        'electron': [-1/2, -1, 1],
        'neutrino': [+1/2, 0, 1],
        'up_type': [+1/2, 2/3, 3],
        'down_type': [-1/2, -1/3, 3],
    }

    T['fermi_coupling'] = siunitx(fermi_coupling)

    widths = {}

    for particle, (i_3, q, n_c) in quantum_numbers.items():
        print()
        print('Particle:', particle)
        print('I_3:', i_3)
        print('Q:', q)
        print('N_color:', n_c)

        g_v = i_3 - 2 * q * sin_sq_weak_mixing
        g_a = i_3

        print('g_v:', g_v)
        print('g_a:', g_a)

        decay_width = n_c / (12 * np.pi) * fermi_coupling \
                * mass_z**3 * (g_a**2 + g_v**2)

        print('Decay width Γ:', decay_width, 'MeV')

        T['gamma_'+particle] = siunitx(decay_width)

        widths[particle] = decay_width

    groups = ['hadronic', 'charged_leptonic', 'neutral_leptonic']

    widths['hadronic'] = 2 * widths['up_type'] + 3 * widths['down_type']
    widths['charged_leptonic'] = 3 * widths['electron']
    widths['neutral_leptonic'] = 3 * widths['neutrino']

    total_width = widths['hadronic'] + widths['charged_leptonic'] + widths['neutral_leptonic']

    ratios = {}

    for group in groups:
        T[group+'_width'] = siunitx(widths[group])
        T['total_width'] = siunitx(total_width)

        ratios[group] = widths[group] / total_width
        T[group+'_ratio'] = siunitx(ratios[group])

        partial_cross_section = 12 * np.pi / mass_z**2 * widths['electron'] * widths[group] / total_width**2
        T[group+'_partial_cross_section'] = siunitx(partial_cross_section / 1e-11)

    total_cross_section = 12 * np.pi / mass_z**2 * widths['electron'] / total_width
    T['total_cross_section'] = siunitx(total_cross_section / 1e-11)


    extra_width = 1 + (widths['up_type'] + widths['down_type'] + widths['charged_leptonic'] + widths['neutral_leptonic']) / total_width
    T['extra_width'] = siunitx(extra_width)


def job_angular_dependence(T):
    x = np.linspace(0.1, np.pi, 100)
    y1 = 1 + np.cos(x)**2
    y2 = 1/(1 - np.cos(x))

    np.savetxt('_build/xy/s-channel.tsv', np.column_stack([x, y1]))
    np.savetxt('_build/xy/t-channel.tsv', np.column_stack([x, y2]))




def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)


def job_radiative_correction(T):
    data = np.loadtxt('Data/radiative_corrections.tsv')
    sqrt_mandelstam_s = data[:, 0]
    correction = data[:, 1]

    pl.plot(sqrt_mandelstam_s, correction)

    interpolator = scipy.interpolate.interp1d(sqrt_mandelstam_s, correction, kind='quadratic')

    x = np.linspace(np.min(sqrt_mandelstam_s), np.max(sqrt_mandelstam_s))
    y = interpolator(x)

    pl.plot(x, y)

    pl.savefig('_build/mpl-radiative.pdf')

    np.savetxt('_build/xy/radiative_data.tsv', np.column_stack([
        sqrt_mandelstam_s, correction
    ]))
    np.savetxt('_build/xy/radiative_interpolated.tsv', np.column_stack([
        x, y,
    ]))

    return interpolator


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

    job_decay_widths(T)
    job_radiative_correction(T)
    job_angular_dependence(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
