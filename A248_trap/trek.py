#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

'''
Importer for TREK oscillosope data.
'''

import argparse
import csv
import os

import numpy as np


def load_file(path):
    '''
    Loads a file from the given path and returns X and Y values.
    '''
    x = []
    y = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[3]))
            y.append(float(row[4]))

    return np.array(x), np.array(y)


def load_dir(digit_str):
    '''
    Loads one or two channels from the data set indicated by a string of
    numbers, say '0001'. Return value are either X, Y or X_1, Y_1, X_2, Y_2. It
    should always be X_1 == X_2 as it is the same image.
    '''
    path = 'Data/ALL{}'.format(digit_str)

    ch1_file = os.path.join(path, 'F{}CH1.CSV'.format(digit_str))
    ch2_file = os.path.join(path, 'F{}CH2.CSV'.format(digit_str))

    has1 = os.path.isfile(ch1_file)
    has2 = os.path.isfile(ch2_file)

    rvalue = []

    if has1:
        x1, y1 = load_file(ch1_file)
        rvalue.append(x1)
        rvalue.append(y1)
    if has2:
        x2, y2 = load_file(ch2_file)
        rvalue.append(x2)
        rvalue.append(y2)

    return rvalue


if __name__ == '__main__':
    print(load_dir('0000'))
