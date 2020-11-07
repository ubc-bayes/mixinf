#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' -d 1 -N 5000 --target 'cauchy' --maxiter 100 -B 1000 --tol 0.00001 --sd 0.1 1 10 100 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
