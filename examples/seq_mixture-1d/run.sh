#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' --sampling 'cont' -d 1 -N 5000 --target 'mixture' --maxiter 10 -B 1000 --tol 0 --sd 0.2 2 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'mixture'

$SHELL
