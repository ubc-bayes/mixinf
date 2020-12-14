#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' --sampling 'lp' -d 2 -N 1000 --target 'banana' --maxiter 20 -B 1000 --tol 0.00001 --sd 1 10 --trace --verbose --profiling

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'banana'

$SHELL
