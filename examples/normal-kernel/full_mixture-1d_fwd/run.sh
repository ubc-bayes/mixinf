#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'full' --forward -d 1 -N 100 --target 'mixture' --maxiter 500 -B 1000 -b 0.0001 --tol 0.01 --sd 0.1 1 10 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'mixture'

$SHELL
