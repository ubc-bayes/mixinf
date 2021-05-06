#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py -N 25 -d 1 --target '4-mixture' --kernel 'gaussian' --rkhs 'rbf' --maxiter 40  -t 50 -T 150 -B 5000 --tol 0.001 --plots --verbose --profiling

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target '4-mixture' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
