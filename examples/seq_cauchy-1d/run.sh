#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' --sampling 'cont' -d 1 -N 1000 --target 'cauchy' --maxiter 10 -B 1000 --tol 0.00001 --sd 1 10 100 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
