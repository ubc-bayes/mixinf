#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'full' -d 2 -N 100 --target 'banana' --maxiter 1000 -B 1000 --tol 0.1 --sd 0.1 1 10 --trace --verbose --profiling

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
