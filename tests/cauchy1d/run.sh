#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -N 10 -d 1 --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf' --maxiter 5  -t 25 -T 500 -B 1000 --tol 0.00001 --weight_max 10 --verbose

# plot
#python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --kernel 'gaussian' --rkhs 'rbf'

$SHELL
