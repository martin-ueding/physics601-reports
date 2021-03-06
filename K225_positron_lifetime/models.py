# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

import numpy as np
import scipy.special


def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)


def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 


def linear(x, a, b):
    return a * x + b


def lifetime_spectrum(t, t0, width, A_0, A_t, tau_0, tau_t, bg):
    summand_1 = _get_summand(A_0, tau_0, width, t, t0)
    summand_2 = _get_summand(A_t, tau_t, width, t, t0)
    return summand_1 + summand_2 + bg


def lifetime_spectrum_parts(t, t0, width, A_0, A_t, tau_0, tau_t, bg):
    summand_1 = _get_summand(A_0, tau_0, width, t, t0)
    summand_2 = _get_summand(A_t, tau_t, width, t, t0)
    return summand_1, summand_2, summand_1 + summand_2 + bg


def _get_exp(sigma, tau, t, t0):
    arg = (sigma**2 - 2 * tau * (t - t0)) / (2 * tau**2)
    return np.exp(arg)


def _get_summand(A, tau, sigma, t, t0):
    prefactor =  A / (2 * tau)
    exponential = _get_exp(sigma, tau, t, t0)
    erf = _get_erf(sigma, tau, t, t0)
    return prefactor * exponential * erf


def _get_a(sigma, tau, t0):
    return (sigma**2 + tau * t0) / (np.sqrt(2) * sigma * tau)


def _get_b(sigma, tau, t, t0):
    return (tau * (t - t0) - sigma**2) / (np.sqrt(2) * sigma * tau)


def _get_erf(sigma, tau, t, t0):
    summand_1 = scipy.special.erf(_get_a(sigma, tau, t0))
    summand_2 = scipy.special.erf(_get_b(sigma, tau, t, t0))
    return summand_1 + summand_2
