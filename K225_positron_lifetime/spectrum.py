# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

'''
Functions for the expected lifetime spectrum.
'''

import glob

import numpy as np
import scipy.optimize as op

import bootstrap
import conf
import models


def job_lifetime_spectra(T):
    files = glob.glob('Data/in-*.txt')

    for i in range(len(files)):
        data = np.loadtxt(files[i])
        channel = data[:,0]
        counts = data[:,1]

        mean = []
        width = []
        A_0 = []
        A_t = []
        tau_0 = []
        tau_t = []
        BG = []

        for a in range(2):
            boot_counts = bootstrap.redraw_count(counts)
            popt, pconv = op.curve_fit(models.lifetime_spectrum, channel, boot_counts, p0=[
                1600,
                45,
                180,
                180,
                40,
                40,
                0
                ])
            mean.append(popt[0])
            width.append(popt[1])
            A_0.append(popt[2])
            A_t.append(popt[3])
            tau_0.append(popt[4])
            tau_t.append(popt[5])
            BG.append(popt[6])

        mean_val, mean_err = bootstrap.average_and_std_arrays(mean)
        width_val, width_err = bootstrap.average_and_std_arrays(width)
        A_0_val, A_0_err = bootstrap.average_and_std_arrays(A_0)
        A_t_val, A_t_err = bootstrap.average_and_std_arrays(A_t)
        tau_0_val, tau_0_err = bootstrap.average_and_std_arrays(tau_0)
        tau_t_val, tau_t_err = bootstrap.average_and_std_arrays(tau_t)
        BG_val, BG_err = bootstrap.average_and_std_arrays(BG)


    x = np.linspace(1000, 3000, 500)
    y = models.lifetime_spectrum(x, mean_val, width_val, A_0_val, A_t_val, tau_0_val, tau_t_val, BG_val)

    pl.plot(channel, counts, linestyle="none", marker="o")
    pl.plot(x, y)
    dandify_plot()
    pl.savefig('_build/mpl-channel-counts.pdf')
    pl.clf()
