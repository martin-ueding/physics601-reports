# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

import matplotlib.pyplot as pl


SAMPLES = 10


def dandify_plot():
    '''
    Common operations to make matplotlib plots look nicer.
    '''
    pl.grid(True)
    pl.margins(0.05)
    pl.tight_layout()
