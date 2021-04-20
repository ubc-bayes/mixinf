#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 -N 25 --kernel 'gaussian' --rkhs 'rbf' --maxiter 30 -t 25 -T 500 --tol 0.00001 --weight_max 10 --bvi_kernels 30 --verbose

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
