#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' --sampling 'cont' -d 1 -N 5000 --target 'cauchy' --maxiter 50 -B 1000 --tol 0.001 --sd 1 10 100 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
