#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' -d 2 -N 1000 --target 'banana' --maxiter 100 -B 1000 --tol 0.001 --sd 0.1 1 10 --trace --verbose --profiling

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
