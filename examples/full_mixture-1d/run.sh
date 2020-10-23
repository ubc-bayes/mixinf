#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'full' -d 1 -N 50 --target 'mixture' --maxiter 1000 -B 1000 --tol 0.1 --sd 0.2 2 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'mixture'

$SHELL
