#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py -N 100 -d 2 --target 'banana-gaussian' --kernel 'gaussian' --rkhs 'rbf' --maxiter 30  -t 25 -T 50 -B 1000 --tol 0.00001 --weight_max 10 --plots --verbose --profiling

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'banana-gaussian' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
