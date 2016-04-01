#!/bin/bash
# Copyright © 2016 Martin Ueding <dev@martin-ueding.de>

for temp in 0222 0510 0663 0771 0887 1025 1244
do
    sed "s/1151/${temp}/g" Plots/lifetime-1151.tex > Plots/lifetime-${temp}.tex
done
