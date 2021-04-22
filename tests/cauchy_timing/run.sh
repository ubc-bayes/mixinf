#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 10 --lbvi -N 25 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 25 -T 500 --tol 0.00001 --weight_max 10 --bvi --bvi_kernels 20 --verbose

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
