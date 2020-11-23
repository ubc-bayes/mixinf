#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'full' --forward -d 1 -N 50 --target 'cauchy' --maxiter 500 -B 1000 --tol 0.1 --sd 0.1 1 10 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
