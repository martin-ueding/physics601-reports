#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import json
import sys

import matplotlib.pyplot as pl
import numpy as np
import scipy.misc
import scipy.ndimage.filters
import scipy.optimize as op
import scipy.stats

from unitprint2 import siunitx
import bootstrap

atomic_unit = 1.6605e-27 # kg
B_val = 33.3 # T
B_err = 1 # T
debye_temp = 470 # K
electron_charge = 1.609e-19 # C
hbar_omega0_ev = 14.4e3 # eV
hbar_omega0_joule = 14.4e3 * electron_charge # J
iron_mass = 56.9353940 * atomic_unit
k_boltzmann = 1.38e-23 # J / K
magneton = 5.05078e-27 # J / T
mu_n = 5.5e-27 # A m^2
room_temp = 292 # K
speed_of_light = 3e8 # m / s

length_val = 25.1e-3
length_err = 0.2e-3

def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)

def lorentz6(x,
             mean1, width1, integral1,
             mean2, width2, integral2,
             mean3, width3, integral3,
             mean4, width4, integral4,
             mean5, width5, integral5,
             mean6, width6, integral6,
             offset):
    return lorentz(x, mean1, width1, integral1) \
            + lorentz(x, mean2, width2, integral2) \
            + lorentz(x, mean3, width3, integral3) \
            + lorentz(x, mean4, width4, integral4) \
            + lorentz(x, mean5, width5, integral5) \
            + lorentz(x, mean6, width6, integral6) \
            + offset

def fit_dip(v, rate_val, rate_err):
    p0_width = 1e-3
    p0_integral = -1e-2
    p0_offset = 58

    popt, pconv = op.curve_fit(lorentz6, v, rate_val, sigma=rate_err,
                               p0=[
                                   -5.3e-3, p0_width, p0_integral,
                                   -3.1e-3, p0_width, p0_integral,
                                   -0.8e-3, p0_width, p0_integral,
                                   0.6e-3, p0_width, p0_integral,
                                   2.9e-3, p0_width, p0_integral,
                                   5.2e-3, p0_width, p0_integral,
                                   p0_offset,
                               ])

    return popt, np.sqrt(pconv.diagonal())


def job_theory(T):
    prefactor_val = speed_of_light * B_val * mu_n / hbar_omega0_joule
    prefactor_err = speed_of_light * B_err * mu_n / hbar_omega0_joule

    debye_1 = 3 * hbar_omega0_joule**2 / (4 * iron_mass * speed_of_light**2 * k_boltzmann * debye_temp)
    debye_2 = 1 + 2 * np.pi**2/3 * (room_temp / debye_temp)**2
    debye_3 = np.exp(- debye_1 * debye_2)

    T['debye_1'] = siunitx(debye_1)
    T['debye_2'] = siunitx(debye_2)
    T['debye_3'] = siunitx(debye_3)
    T['iron_mass'] = siunitx(iron_mass)
    T['room_temp'] = siunitx(room_temp)

    T['length_mm'] = siunitx(length_val / 1e-3, length_err / 1e-3)

    T['v_prefactor'] = siunitx(prefactor_val, prefactor_err)
    T['B'] = siunitx(B_val, B_err)
    T['hbar_omega0_joule'] = siunitx(hbar_omega0_joule)
    T['hbar_omega0_ev'] = siunitx(hbar_omega0_ev)


def job_spectrum(T):
    data = np.loadtxt('Data/runs.tsv')


    T['raw_table'] = list([[str(int(x)) for x in row] for row in data])

    runs = data[:, 0]
    motor = data[:, 1]
    T_LR = data[:, 2]
    N_LR = data[:, 3]
    T_RL = data[:, 4]
    N_RL = data[:, 5]

    time_lr = T_LR * 10e-3
    time_rl = T_RL * 10e-3

    velocity_lr_val = - length_val * runs / time_lr
    velocity_rl_val = length_val * runs / time_rl
    velocity_lr_err = - length_err * runs / time_lr
    velocity_rl_err = length_err * runs / time_rl

    rate_lr_val = N_LR / time_lr
    rate_rl_val = N_RL / time_rl
    rate_lr_err = np.sqrt(N_LR) / time_lr
    rate_rl_err = np.sqrt(N_RL) / time_rl

    rate_val = np.concatenate((rate_lr_val, rate_rl_val))
    rate_err = np.concatenate((rate_lr_err, rate_rl_err))
    velocity_val = np.concatenate((velocity_lr_val, velocity_rl_val))
    velocity_err = np.concatenate((velocity_lr_err, velocity_rl_err))

    fit_val, fit_err = fit_dip(velocity_val, rate_val, rate_err)
    x = np.linspace(np.min(velocity_val), np.max(velocity_val), 1000)
    y = lorentz6(x, *fit_val)
    pl.plot(x, y)

    widths_val = fit_val[1::3]
    widths_err = fit_err[1::3]

    delta_e_val = hbar_omega0_ev * widths_val / speed_of_light
    delta_e_err = hbar_omega0_ev * widths_err / speed_of_light

    relative_width_val = widths_val / speed_of_light
    relative_width_err = widths_err / speed_of_light

    T['widths_table'] = list(zip(*[
        siunitx(widths_val / 1e-6, widths_err / 1e-6),
        siunitx(delta_e_val / 1e-9, delta_e_err / 1e-9),
        siunitx(relative_width_val / 1e-13, relative_width_err / 1e-13),
    ]))

    formatted = siunitx(fit_val / 1e-6, fit_err / 1e-6)
    offset = siunitx(fit_val[-1], fit_err[-1])

    T['fit_param'] = list(zip(*[iter(formatted[:-1])]*3))
    T['fit_offset'] = offset

    np.savetxt('_build/xy/rate_fit.csv', np.column_stack([
        x / 1e-3, y
    ]))

    np.savetxt('_build/xy/rate_lr.csv', np.column_stack([
        velocity_lr_val / 1e-3, rate_lr_val, rate_lr_err,
    ]))
    np.savetxt('_build/xy/rate_rl.csv', np.column_stack([
        velocity_rl_val / 1e-3, rate_rl_val, rate_rl_err,
    ]))

    np.savetxt('_build/xy/motor_lr.csv', np.column_stack([
        motor, -velocity_lr_val / 1e-3, velocity_lr_err / 1e-3,
    ]))
    np.savetxt('_build/xy/motor_rl.csv', np.column_stack([
        motor, velocity_rl_val / 1e-3, velocity_rl_err / 1e-3,
    ]))

    pl.errorbar(velocity_lr_val, rate_lr_val, rate_lr_err, xerr=velocity_lr_err, marker='o', linestyle='none')
    pl.errorbar(velocity_rl_val, rate_rl_val, rate_lr_err,  xerr=velocity_rl_err, marker='o', linestyle='none')
    pl.grid(True)
    pl.margins(0.05)
    pl.tight_layout()
    pl.savefig('_build/mpl-rate.pdf')
    pl.clf()

    #pl.errorbar(motor, -velocity_lr_val, velocity_lr_err, marker='o')
    #pl.errorbar(motor,  velocity_rl_val, velocity_rl_err, marker='o')
    #pl.grid(True)
    #pl.margins(0.05)
    #pl.tight_layout()
    #pl.savefig('_build/mpl-motor.pdf')
    #pl.clf()

    means_val = fit_val[:-1][0::3]
    means_err = fit_err[:-1][0::3]

    lande_factors(T, means_val, means_err)


def lande_factors(T, centers_val, centers_err):
    isomeric_val = np.mean(centers_val)
    isomeric_err = np.sqrt(np.mean(centers_err**2))

    factor_val = speed_of_light * B_val * magneton / hbar_omega0_joule
    factor_err = speed_of_light * B_err * magneton / hbar_omega0_joule

    T['isomeric_mm_s'] = siunitx(isomeric_val / 1e-3, isomeric_err / 1e-3)
    T['isomeric_nev'] = siunitx(hbar_omega0_ev * isomeric_val / speed_of_light / 1e-9, hbar_omega0_ev * isomeric_err / speed_of_light / 1e-9)

    if True:
        v_shift_e_val = - np.mean([
            #centers_val[2] - centers_val[0],
            centers_val[3] - centers_val[1],
            centers_val[4] - centers_val[2],
            #centers_val[5] - centers_val[3],
        ])
        v_shift_e_err = np.std([
            #centers_val[2] - centers_val[0],
            centers_val[3] - centers_val[1],
            centers_val[4] - centers_val[2],
            #centers_val[5] - centers_val[3],
        ])
        v_shift_e_err = np.sqrt(np.mean([
            centers_err[1]**2,
            centers_err[2]**2,
            centers_err[3]**2,
            centers_err[4]**2,
        ]))

        v_shift_q_val = ((centers_val[5] - centers_val[3]) - (centers_val[2] - centers_val[0]))/2
        v_shift_q_err = np.sqrt(
            centers_err[0]**2 + centers_err[2]**2 + centers_err[3]**2 + centers_err[5]**2
        ) / 2

        v_shift_g_val = np.mean([
            centers_val[2] - centers_val[1],
            centers_val[4] - centers_val[3],
        ])
        v_shift_g_err = np.std([
            centers_val[2] - centers_val[1],
            centers_val[4] - centers_val[3],
        ])
    else:
        v_shift_e_val = - np.mean([
            centers_val[1] - centers_val[0],
            centers_val[2] - centers_val[1],
            centers_val[4] - centers_val[3],
            centers_val[5] - centers_val[4],
        ])
        v_shift_e_err = np.std([
            centers_val[1] - centers_val[0],
            centers_val[2] - centers_val[1],
            centers_val[4] - centers_val[3],
            centers_val[5] - centers_val[4],
        ])
        #v_shift_e_err = np.sqrt(np.mean(centers_err**2))

        v_shift_g_val = np.mean([
            centers_val[4] - centers_val[0],
            centers_val[5] - centers_val[1],
        ])
        v_shift_g_err = np.std([
            centers_val[4] - centers_val[0],
            centers_val[5] - centers_val[1],
        ])

    lande_e_val = v_shift_e_val / factor_val
    lande_g_val = v_shift_g_val / factor_val
    lande_e_err = np.sqrt((v_shift_e_err / factor_val)**2 + (v_shift_e_val / factor_val**2 * factor_err)**2)
    lande_g_err = np.sqrt((v_shift_g_err / factor_val)**2 + (v_shift_g_val / factor_val**2 * factor_err)**2)

    T['v_shift_g'] = siunitx(v_shift_g_val / 1e-3, v_shift_g_err / 1e-3)
    T['v_shift_e'] = siunitx(v_shift_e_val / 1e-3, v_shift_e_err / 1e-3)
    T['v_shift_q'] = siunitx(v_shift_q_val / 1e-3, v_shift_q_err / 1e-3, error_digits=2)
    T['lande_g'] = siunitx(lande_g_val, lande_g_err)
    T['lande_e'] = siunitx(lande_e_val, lande_e_err)


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

    job_spectrum(T)
    job_theory(T)

    test_keys(T)
    with open('_build/template.js', 'w') as f:
        json.dump(dict(T), f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
