#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

import argparse
import random

import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as op


def main():
    options = _parse_args()

    numbers = [random.gauss(3, 1) for i in range(10000)]

    std = np.std(numbers)
    print('std', std)

    r_up = 50 + 68.3/2
    r_down = 50 - 68.3/2

    p_up = np.percentile(numbers, r_up)
    p_down = np.percentile(numbers, r_down)
    median = np.median(numbers)

    print(p_down, p_up, (p_up - p_down)/2)

    e_up = p_up - median 
    e_down = median - p_down

    print(e_up, e_down)


def _parse_args():
    '''
    Parses the command line arguments.

    :return: Namespace with arguments.
    :rtype: Namespace
    '''
    parser = argparse.ArgumentParser(description='')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
