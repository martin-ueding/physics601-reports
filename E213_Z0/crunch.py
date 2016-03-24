#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

# TODO Rename Tau to Tauon everywhere

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
import mpl_toolkits.mplot3d.axes3d as p3

from unitprint2 import siunitx
import bootstrap

fermi_coupling = 1.6637e-11 # MeV^{-2}
mass_z = 91182 # MeV
sin_sq_weak_mixing = 0.2312
weak_mixing_angle = np.arcsin(np.sqrt(sin_sq_weak_mixing))

default_figsize = (15.1 / 2.54, 8.3 / 2.54)

names = ['electron', 'muon', 'tau', 'hadron']

energies = np.loadtxt('Data/energies.txt')

channel_colors = [
    '#377eb8',
    '#984ea3',
    '#4daf4a',
    '#e41a1c',
]

pp = pprint.PrettyPrinter()

def bootstrap_kernel(mc_sizes, matrix, readings, lum, radiative_hadrons,
                     radiative_leptons):
    '''
    Core of the analysis.

    Everything that is done here does not care about errors at all. The errors
    only emerge via a lot (> 100) runs of this bootstrap kernel with slightly
    different input values.

    :param np.array mc_sizes: List of raw number of MC events, four entries
    :param np.array matrix: Detection matrix, 4×4 numbers
    :param list(np.array) readings: For each energy, there is a reading 4-vector
    :param np.array lum_val: Luminosity for the seven energies
    '''
    # Normalize the raw_matrix.
    matrix = matrix.dot(np.diag(1/mc_sizes))

    inverted = np.linalg.inv(matrix)

    corr_list = []

    for i in range(7):
        vector = readings[i, :]
        corrected = inverted.dot(vector)

        corr_list.append(corrected)

    corr = np.column_stack(corr_list)

    masses = []
    widths = []
    cross_sections = []
    y_list = []

    x = np.linspace(np.min(energies), np.max(energies), 200)

    for i, name in zip(range(len(names)), names):
        counts = corr[i, :]
        cross_section = counts / lum

        # Radiative corrections for cross section.
        if i == 3:
            cross_section += radiative_hadrons
        else:
            cross_section += radiative_leptons

        # Add the seven cross sections for this type to the list of cross
        # sections.
        cross_sections.append(cross_section)

        # Fit the curve, add the fit parameters to the lists.
        popt, pconv = op.curve_fit(lorentz, energies, cross_section)
        masses.append(popt[0])
        widths.append(popt[1])

        # Sample the fitted curve, add to the list.
        y = lorentz(x, *popt)
        y_list.append(y)

    return x, masses, widths, np.array(cross_sections), y_list, corr.T, \
            matrix, inverted, readings


def bootstrap_driver(T):
    # Load all the input data from the files.
    lum_data = np.loadtxt('Data/luminosity.txt')
    lum_val = lum_data[:, 0]
    lum_err = lum_data[:, 3]
    radiative_hadrons = np.loadtxt('Data/radiative-hadrons.tsv')
    radiative_leptons = np.loadtxt('Data/radiative-leptons.tsv')
    raw_matrix = np.loadtxt('Data/matrix.txt').T
    mc_sizes = np.loadtxt('Data/monte-carlo-sizes.txt')
    filtered = np.loadtxt('Data/filtered.txt')

    # Some output into the template.
    T['luminosities_table'] = list(zip(siunitx(energies), siunitx(lum_val, lum_err)))
    T['radiative_cs_table'] = list(zip(
        siunitx(energies),
        siunitx(radiative_hadrons),
        siunitx(radiative_leptons),
    ))

    # Container for the results of each bootstrap run.
    results = []

    for r in range(100):
        # Draw new numbers for the matrix.
        boot_matrix = redraw_count(raw_matrix)

        # Draw new luminosities.
        boot_lum_val = np.array([
            random.gauss(val, err)
            for val, err
            in zip(lum_val, lum_err)])

        # Draw new filtered readings.
        boot_readings = redraw_count(filtered)

        # Run the analysis on the resampled data and save the results.
        results.append(bootstrap_kernel(mc_sizes, boot_matrix, boot_readings,
                                        boot_lum_val, radiative_hadrons,
                                        radiative_leptons))

    # The `results` is a list which contains one entry per bootstrap run. This
    # is not particularly helpful as the different interesting quantities are
    # only on the second index on the list. The first index of the `results`
    # list is the bootstrap run index. Therefore we use the `zip(*x)` trick to
    # exchange the two indices. The result will be a list of quantities which
    # are themselves lists of the bootstrap samples. Then using Python tuple
    # assignments, we can split that (now) outer list into different
    # quantities. Each of the new variables created here is a list of R
    # bootstrap samples.
    x_dist, masses_dist, widths_dist, cross_sections_dist, y_dist, corr_dist, \
            matrix_dist, inverted_dist, readings_dist = zip(*results)

    # We only need one of the lists of the x-values as they are all the same.
    # So take the first and throw the others out.
    x = x_dist[0]

    # The masses and the widths that are given back from the `bootstrap_kernel`
    # are a list of four elements (electrons, muons, tauons, hadrons) each. The
    # variable `masses_dist` contains R copies of this four-list, one copy for
    # each bootstrap sample. We now average along the bootstrap dimension, that
    # is the outermost dimension. For each of the four masses, we take the
    # average along the R copies. This will give us four masses and four
    # masses-errors.
    masses_val, masses_err = bootstrap.average_and_std_arrays(masses_dist)
    widths_val, widths_err = bootstrap.average_and_std_arrays(widths_dist)

    # Format masses and widths for the template.
    T['lorentz_fits_table'] = list(zip(
        [name.capitalize() for name in names],
        siunitx(masses_val, masses_err),
        siunitx(widths_val, widths_err, error_digits=2),
    ))

    # Format original counts for the template.
    val, err = bootstrap.average_and_std_arrays(readings_dist)
    T['counts_table'] = []
    for i in range(7):
        T['counts_table'].append([siunitx(energies[i])] + siunitx(val[i, :], err[i, :], allowed_hang=10))

    # Format corrected counts for the template.
    val, err = bootstrap.average_and_std_arrays(corr_dist)
    T['corrected_counts_table'] = []
    for i in range(7):
        T['corrected_counts_table'].append([siunitx(energies[i])] + siunitx(val[i, :], err[i, :], allowed_hang=10))

    # Format matrix for the template.
    matrix_val, matrix_err = bootstrap.average_and_std_arrays(matrix_dist)
    T['matrix'] = []
    for i in range(4):
        T['matrix'].append([names[i].capitalize()] + siunitx(matrix_val[i, :]*100, matrix_err[i, :]*100, allowed_hang=10))

    # Format inverted matrix for the template.
    inverted_val, inverted_err = bootstrap.average_and_std_arrays(inverted_dist)
    T['inverted'] = []
    for i in range(4):
        T['inverted'].append([names[i].capitalize()+'s'] +
                             list(map(number_padding,
                             siunitx(inverted_val[i, :], inverted_err[i, :], allowed_hang=10))))

    # Format cross sections for the template.
    cs_val, cs_err = bootstrap.average_and_std_arrays(cross_sections_dist)
    T['cross_sections_table'] = []
    for i in range(7):
        T['cross_sections_table'].append([siunitx(energies[i])] + siunitx(cs_val[:, i], cs_err[:, i]))

    # Build error band for pgfplots.
    y_list_val, y_list_err = bootstrap.average_and_std_arrays(y_dist)
    for i, name in zip(itertools.count(), names):
        # Extract the y-values for the given decay type.
        y_val = y_list_val[i, :]
        y_err = y_list_err[i, :]

        # Store the data for pgfplots.
        np.savetxt('_build/xy/cross_section-{}s.tsv'.format(name),
                   np.column_stack([energies, cs_val[i, :], cs_err[i, :]]))
        np.savetxt('_build/xy/cross_section-{}s-band.tsv'.format(name),
                   bootstrap.pgfplots_error_band(x, y_val, y_err))


def number_padding(number):
    if number[0] == '-':
        return r'$\num{'+number+'}$'
    else:
        return r'$\phantom{-}\num{'+number+'}$'


def redraw_count(a):
    '''
    Takes a ``np.array`` with counts and re-draws the counts from the implicit
    Gaussian distribution with width ``sqrt(N)``.
    '''
    out = [random.gauss(x, np.sqrt(x)) for x in a]
    return np.array(out).reshape(a.shape)


def visualize_matrix(matrix, name):
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


def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)


def job_colors():
    colors = [(55,126,184), (152,78,163), (77,175,74), (228,26,28)]

    with open('_build/colors.tex', 'w') as f:
        for name, color in zip(names, colors):
            f.write(r'\definecolor{{{}s}}{{rgb}}{{{},{},{}}}'.format(name, *[x/255 for x in color]) + '\n')


def figname(basename):
    return '_build/to_crop/mpl-{}.pdf'.format(basename)


def job_afb_analysis(T, interpolator):
    # TODO Remove interpolator, use direct energies.
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

    # TODO Extract sin_sq.

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

        hist, edges = np.histogram(ctrk_n, bins=log_bins)
        hist_extended = np.array(list(hist) + [hist[-1]])
        np.savetxt('_build/xy/hist-ctrk_n-'+file_+'.tsv',
                   np.column_stack([edges, hist_extended]))

        hist, edges = np.histogram(ctrk_sump)
        hist_extended = np.array(list(hist) + [hist[-1]])
        np.savetxt('_build/xy/hist-ctrk_sump-'+file_+'.tsv',
                   np.column_stack([edges, hist_extended]))

        hist, edges = np.histogram(ecal_sume)
        hist_extended = np.array(list(hist) + [hist[-1]])
        np.savetxt('_build/xy/hist-ecal_sume-'+file_+'.tsv',
                   np.column_stack([edges, hist_extended]))

        hist, edges = np.histogram(hcal_sume)
        hist_extended = np.array(list(hist) + [hist[-1]])
        np.savetxt('_build/xy/hist-hcal_sume-'+file_+'.tsv',
                   np.column_stack([edges, hist_extended]))

        # TODO Better angle.
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

    bootstrap_driver(T)
    job_decay_widths(T)
    job_colors()
    job_afb_analysis(T, interpolator)
    job_grope(T, options.show)
    job_angular_dependence(T)
    job_asymetry(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
