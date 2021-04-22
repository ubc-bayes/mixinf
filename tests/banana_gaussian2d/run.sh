#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'banana-gaussian' -B 1000 --lbvi -N 100 --kernel 'gaussian' --rkhs 'rbf' --maxiter 30 -t 25 -T 50 --tol 0.00001 --weight_max 10 --bvi_kernels 2 --verbose

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'banana-gaussian' --kernel 'gaussian' --rkhs 'rbf'

$SHELL