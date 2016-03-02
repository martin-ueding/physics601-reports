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

atomic_unit = 1.6605e-27 # kg
B_err = 1 # T
B_val = 33.3 # T
debye_temp = 470 # K
electron_charge = 1.609e-19 # C
hbar_omega0_ev = 14.4e3 # eV
hbar_omega0_joule = 14.4e3 * electron_charge # J
iron_mass = 56.9353940 * atomic_unit
k_boltzmann = 1.38e-23 # J / K
mu_n = 5.5e-27 # A m^2
room_temp = 292 # K
speed_of_light = 3e8 # m / s

length_val = 25.1e-3
length_err = 0.2e-3

def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)

def lorentz4(x,
             mean1, width1, integral1,
             mean2, width2, integral2,
             offset):
    return lorentz(x, mean1, width1, integral1) \
            + lorentz(x, mean2, width2, integral2) \
            + offset


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

    T['v_prefactor'] = siunitx(prefactor_val, prefactor_err)
    T['B'] = siunitx(B_val, B_err)
    T['hbar_omega0_joule'] = siunitx(hbar_omega0_joule)
    T['hbar_omega0_ev'] = siunitx(hbar_omega0_ev)


def fit_dip(v, rate_val, rate_err):
    selection = (min < v) & (v < max)
    v = v[selection]
    rate_val = rate_val[selection]
    rate_err = rate_err[selection]

    popt, pconv = op.curve_fit(lorentz, v, rate_val, sigma=rate_err,
                               p0=[(min+max)/2, 1e-3, -1e-2, 60])

    x = np.linspace(np.min(v), np.max(v), 1000)
    y = lorentz( x, *popt)

    return popt, x, y


def job_spectrum(T):
    data = np.loadtxt('Data/runs.tsv')

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

    popt, x, y = fit_dip(velocity_val, rate_val, rate_err, 4e-3, 6.5e-3)
    print(popt)
    pl.plot(x, y)

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
