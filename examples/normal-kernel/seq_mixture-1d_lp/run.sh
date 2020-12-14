#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' --sampling 'lp' -d 1 -N 1000 --target 'mixture' --maxiter 50 -B 1000 --tol 0.00001 --sd 0.2 2 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'mixture'

$SHELL
