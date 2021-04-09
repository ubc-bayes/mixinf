#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py -N 10 -d 2 --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf' --maxiter 50  -t 25 -T 500 -B 1000 --tol 0.00001 --weight_max 10 --plots --verbose --profiling

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
