#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' --sampling 'lp' -d 2 -N 1000 --target 'mixture' --maxiter 30 -B 10000 --tol 0 --sd 0.2 2 --trace --verbose --profiling

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'mixture'

$SHELL
