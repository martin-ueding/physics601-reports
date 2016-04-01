# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

import glob
import itertools
import os
import pprint
import re

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl

import bootstrap
import conf
import models
import spectrum


pp = pprint.PrettyPrinter()


def job_temperature_dependence(T, indium_spectra):
    temps_lower, temps_upper, taus_0_val, taus_0_err, taus_t_val, taus_t_err = map(np.array, indium_spectra)

    pl.clf()
    pl.errorbar((temps_lower + temps_upper)/2, taus_0_val, yerr=taus_0_err, linestyle='none', marker='+')
    pl.xlabel('Mean temp')
    pl.ylabel(r'$\tau_0$')
    conf.dandify_plot()
    pl.savefig('_build/mpl-tau_0.pdf')
    pl.clf()


    pl.errorbar((temps_lower + temps_upper)/2, taus_t_val, yerr=taus_t_err, linestyle='none', marker='+')
    pl.xlabel('Mean temp')
    pl.ylabel(r'$\tau_\mathrm{t}$')
    conf.dandify_plot()
    pl.savefig('_build/mpl-tau_t.pdf')
    pl.clf()

