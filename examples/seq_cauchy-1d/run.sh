#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py -d 1 -N 10 --target 'cauchy' -B 1000 --tol 0.01 --trace --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'cauchy'

$SHELL
