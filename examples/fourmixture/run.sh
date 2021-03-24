#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py -N 2 -d 1 --target '4-mixture' --kernel 'gaussian' --rkhs 'rbf' --maxiter 10  -t 25 -T 500 -B 1000 --tol 0.001 --plots --verbose --profiling

# plot
#python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --disc 'kl' --target 'mixture'

$SHELL
