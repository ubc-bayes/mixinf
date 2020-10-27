#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' -d 1 -N 1000 --target 'cauchy' --maxiter 100 -B 1000 --tol 0.01 --sd 1 10 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
