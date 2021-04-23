#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'banana-gaussian' -B 1000 -N 100 --kernel 'gaussian' --rkhs 'rbf' --maxiter 30 -t 25 -T 50 --tol 0.00001 --weight_max 10 --bvi --bvi_kernels 50 --verbose

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'banana-gaussian' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
