#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py -N 20 -d 1 --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf' --maxiter 100  -t 25 -T 500 -B 1000 --tol 0.001 --plots --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
