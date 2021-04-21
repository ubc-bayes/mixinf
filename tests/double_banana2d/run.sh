#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'double-banana' -B 1000 --lbvi -N 50 --kernel 'gaussian' --rkhs 'rbf' --maxiter 20 -t 25 -T 500 --tol 0.00001 --weight_max 10 --bvi --bvi_kernels 20 --verbose

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
