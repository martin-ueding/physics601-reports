# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

import numpy as np


def lorentz(x, mean, width, integral):
    return integral/np.pi * (width/2) / ((x - mean)**2 + (width/2)**2)


def gauss(x, mean, sigma, a):
    return a / (np.sqrt(2 * np.pi) * sigma) \
            * np.exp(- (x - mean)**2 / (2 * sigma**2)) 


def linear(x, a, b):
    return a * x + b
