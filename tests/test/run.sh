#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target '4-mixture' -B 10000 --reps 1 --tol 0.0 --stop 'median' --lbvi -N 10 --kernel 'gaussian' --rkhs 'rbf' --maxiter 20 -t 25 -T 500 --weight_max 20 --cleanup --verbose



# plot
#python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 10 --tol 0.1 0.05 0.01 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --rwmh

$SHELL
