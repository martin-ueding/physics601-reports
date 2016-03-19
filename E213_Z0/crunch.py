#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import json
import os
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

fermi_coupling = 1.6637e-11 # MeV^{-2}
mass_z = 91182 # MeV
sin_sq_weak_mixing = 0.2312
weak_mixing_angle = np.arcsin(np.sqrt(sin_sq_weak_mixing))

default_figsize = (15.1 / 2.54, 8.3 / 2.54)

names = ['electron', 'muon', 'tau', 'hadron']


def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)


def job_colors():
    colors = [(55,126,184), (152,78,163), (77,175,74), (228,26,28)]

    with open('_build/colors.tex', 'w') as f:
        for name, color in zip(names, colors):
            f.write(r'\definecolor{{{}s}}{{rgb}}{{{},{},{}}}'.format(name, *[x/255 for x in color]) + '\n')


def job_cross_sections(T):
    inverse_val, inverse_err = matrix(T)

    filtered = np.loadtxt('Data/filtered.txt')
    energies = np.loadtxt('Data/energies.txt')

    lum_data = np.loadtxt('Data/luminosity.txt')
    lum_val = lum_data[:, 0]
    lum_err = lum_data[:, 3]

    T['luminosities_table'] = list(zip(siunitx(energies), siunitx(lum_val, lum_err)))

    corr_list = []

    for i in range(7):
        vector = filtered[i, :]
        corrected_val = inverse_val.dot(vector)

        corr_list.append(corrected_val)

    corr = np.column_stack(corr_list)

    print('Corr')
    print(corr)

    masses = {}
    widths = {}

    table_counts = []
    table_cross_sections = []

    for i, name in zip(range(len(names)), names):
        counts = corr[i, :]
        cross_section_val = counts / lum_val
        cross_section_err = counts / lum_val**2 * lum_err

        np.savetxt('_build/xy/cross_section-{}s.tsv'.format(name), np.column_stack([energies, cross_section_val, cross_section_err]))

        popt, pconv = op.curve_fit(lorentz, energies, cross_section_val, sigma=cross_section_err)

        x = np.linspace(np.min(energies), np.max(energies), 500)
        y = lorentz(x, *popt)
        np.savetxt('_build/xy/cross_section-{}s-fit.tsv'.format(name), np.column_stack([x, y]))

        perr = np.sqrt(pconv.diagonal())

        masses[name] = popt[0], perr[0]
        widths[name] = popt[1], perr[1]

        table_counts.append(map(str, map(int, counts)))
        table_cross_sections.append(siunitx(cross_section_val, cross_section_err))

    T['lorentz_fits_table'] = []

    for name in names:
        T['lorentz_fits_table'].append([name.capitalize(), siunitx(*masses[name]), siunitx(*widths[name])])

    T['counts_table'] = list(zip(siunitx(energies), *[map(str, map(int, row)) for row in filtered.T]))
    T['corrected_counts_table'] = list(zip(siunitx(energies), *table_counts))
    T['cross_sections_table'] = list(zip(siunitx(energies), *table_cross_sections))
        

def figname(basename):
    return '_build/to_crop/mpl-{}.pdf'.format(basename)


def matrix(T):
    '''
    Generate the inverse mixing matrix and corresponding error matrix.
    '''
    raw_matrix = np.loadtxt('Data/matrix.txt').T
    mc_sizes = np.loadtxt('Data/monte-carlo-sizes.txt')

    print(raw_matrix)
    print(np.diag(1/mc_sizes))

    # Normalize the raw_matrix.
    matrix = raw_matrix.dot(np.diag(1/mc_sizes))

    fig = pl.figure(figsize=default_figsize)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(matrix, cmap='Greens', interpolation='nearest')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels([r'{} $\to$'.format(name) for name in names], rotation=20)
    ax.set_yticklabels([r'$\to$ {}'.format(name) for name in names])
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figname('normalized_matrix'))

    inverted = np.linalg.inv(matrix)

    fig = pl.figure(figsize=default_figsize)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(inverted, cmap='Greens', interpolation='nearest')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$\to$ {}'.format(name) for name in names], rotation=20)
    ax.set_yticklabels([r'{} $\to$'.format(name) for name in names])
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figname('inverted_matrix'))

    # FIXME Actual error matrix
    return inverted, inverted


def job_afb_analysis(T, interpolator):
    energies = np.loadtxt('Data/energies.txt')
    data = np.loadtxt('Data/afb.txt')
    negative = data[:, 0]
    positive = data[:, 1]

    fig = pl.figure(figsize=default_figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(energies, negative)
    ax.plot(energies, positive)
    fig.savefig(figname('afb_raw'))

    afb_val = (positive - negative) / (positive + negative)
    afb_err = np.sqrt(
        (2 * negative / (positive + negative)**2 * np.sqrt(positive))**2
        + (2 * positive / (positive + negative)**2 * np.sqrt(negative))**2
    )

    np.savetxt('_build/xy/afb.tsv', np.column_stack([energies, afb_val, afb_err]))

    afb_corr_val = afb_val + interpolator(energies)

    np.savetxt('_build/xy/afb_corr.tsv', np.column_stack([energies, afb_corr_val, afb_err]))


def job_grope(T, show=False):
    files = ['electrons',
             'muons',
             'quarks',
             'tauons']
    colors = iter([
        '#377eb8',
        '#984ea3',
        '#e41a1c',
        '#4daf4a',
    ])

    fig_3d = pl.figure()
    ax_3d = p3.Axes3D(fig_3d)

    fig = pl.figure(figsize=(12, 8))
    ax_n = fig.add_subplot(2, 2, 1)
    ax_sump = fig.add_subplot(2, 2, 2)
    ax_ecal = fig.add_subplot(2, 2, 3)
    ax_hcal = fig.add_subplot(2, 2, 4)

    log_bins = np.logspace(0, 2, 20)

    for file_ in files:
        data = np.loadtxt(os.path.join('Data', file_ + '.txt'), usecols=(0, 1, 2, 3))

        ctrk_n = data[:, 0]
        ctrk_sump = data[:, 1]
        ecal_sume = data[:, 2]
        hcal_sume = data[:, 3]

        color = next(colors)

        options = {
            'label': file_,
            'color': color,
            'edgecolor': color,
            'alpha': 0.6,
        }

        ax_n.hist(ctrk_n, bins=log_bins, **options)
        ax_sump.hist(ctrk_sump, **options)
        ax_ecal.hist(ecal_sume, **options)
        ax_hcal.hist(hcal_sume, **options)

        ax_3d.scatter(
            ctrk_n,
            #hcal_sume,
            ctrk_sump, ecal_sume, marker="o", color=color, label=file_, s=80)

    ax_n.set_xscale('log')
    ax_n.set_xlabel('Ctrk(N)')
    ax_sump.set_xlabel('Ctrk(Sump)')
    ax_ecal.set_xlabel('Ecal(SumE)')
    ax_hcal.set_xlabel('Hcal(SumE)')

    ax_3d.set_xlabel('Ctrk(N)')
    #ax_3d.set_xlabel('Hcal(SumE)')
    ax_3d.set_ylabel('Ctrk(Sump)')
    ax_3d.set_zlabel('Ecal(SumE)')
    ax_3d.legend(loc='best')

    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        ax.legend(loc='best')
        ax.margins(0.05)


    fig.tight_layout()
    fig.savefig(figname('hist'))

    if show:
        fig_3d.show()
        input()

    fig_3d.savefig(figname('scatter'))

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
        g_v = i_3 - 2 * q * sin_sq_weak_mixing
        g_a = i_3

        decay_width = n_c / (12 * np.pi) * fermi_coupling \
                * mass_z**3 * (g_a**2 + g_v**2)

        T['gamma_'+particle] = siunitx(decay_width)

        widths[particle] = decay_width

        if False:
            print()
            print('Particle:', particle)
            print('I_3:', i_3)
            print('Q:', q)
            print('N_color:', n_c)
            print('g_v:', g_v)
            print('g_a:', g_a)
            print('Decay width Γ:', decay_width, 'MeV')

    groups = ['hadronic', 'charged_leptonic', 'neutral_leptonic']

    widths['hadronic'] = 2 * widths['up_type'] + 3 * widths['down_type']
    widths['charged_leptonic'] = 3 * widths['electron']
    widths['neutral_leptonic'] = 3 * widths['neutrino']

    global total_width
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
    x = np.linspace(-0.9, 0.9, 100)
    y1 = 1 + x**2
    y2 = 1/(1 - x)

    np.savetxt('_build/xy/s-channel.tsv', np.column_stack([x, y1]))
    np.savetxt('_build/xy/t-channel.tsv', np.column_stack([x, y2]))
    np.savetxt('_build/xy/s-t-channel.tsv', np.column_stack([x, y1+y2]))


def job_asymetry(T):
    s_array = np.array([91.225, 89.225, 93.225])
    sin_sq_array = np.array([0.21, 0.23, 0.25])
    q = -1
    i_3 = -1/2

    angle_array = np.arcsin(np.sqrt(sin_sq_array))
    #re_propagator = 1 / (s_array - (mass_z/1000)**2)
    re_propagator = s_array * (s_array - (mass_z/1000)**2) / \
            ((s_array - (mass_z/1000)**2)**2 + s_array * total_width / (mass_z/1000))

    a_e = i_3 / (2 * np.sin(angle_array) * np.cos(angle_array))
    a_f = a_e
    v_e = (i_3 - 2 * q * sin_sq_array) / (2 * np.sin(angle_array) * np.cos(angle_array))
    v_f = v_e

    asymmetry = np.outer(re_propagator, -3/2 * a_e * a_f * q / ((v_e**2 + a_e**2) * (a_f**2 + a_f**2)))

    print(asymmetry)

    T['asymmetry_table'] = list([[siunitx(s)] + siunitx(row, digits=5) for s, row in zip(s_array, asymmetry)])
    T['sin_sq_array'] = siunitx(sin_sq_array)
    T['s_array'] = siunitx(s_array)



    s_array = np.linspace(88.4, 93.8, 200)
    re_propagator = s_array * (s_array - (mass_z/1000)**2) / \
            ((s_array - (mass_z/1000)**2)**2 + s_array * total_width / (mass_z/1000))
    a_e = i_3 / (2 * np.sin(weak_mixing_angle) * np.cos(weak_mixing_angle))
    a_f = a_e
    v_e = (i_3 - 2 * q * sin_sq_weak_mixing) / (2 * np.sin(weak_mixing_angle) * np.cos(weak_mixing_angle))
    v_f = v_e
    asymmetry = re_propagator * (-3)/2 * a_e * a_f * q / ((v_e**2 + a_e**2) * (a_f**2 + a_f**2))

    np.savetxt('_build/xy/afb_theory.tsv', np.column_stack([s_array, asymmetry]))



def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)


def job_radiative_correction(T):
    data = np.loadtxt('Data/radiative_corrections.tsv')
    sqrt_mandelstam_s = data[:, 0]
    correction = data[:, 1]

    pl.clf()
    pl.plot(sqrt_mandelstam_s, correction)

    sqrt_mandelstam_s[0] -= 1e-2
    sqrt_mandelstam_s[-1] += 1e-2

    interpolator = scipy.interpolate.interp1d(sqrt_mandelstam_s, correction, kind='quadratic')

    x = np.linspace(np.min(sqrt_mandelstam_s), np.max(sqrt_mandelstam_s))
    y = interpolator(x)

    pl.plot(x, y)

    pl.savefig(figname('radiative'))

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    options = parser.parse_args()

    interpolator = job_radiative_correction(T)

    job_colors()
    job_cross_sections(T)
    job_afb_analysis(T, interpolator)
    job_grope(T, options.show)
    job_decay_widths(T)
    job_angular_dependence(T)
    job_asymetry(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
