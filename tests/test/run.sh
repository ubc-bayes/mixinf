#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10 --reps 6 7 8 9 10 --tol 0.0 --stop 'median' --gvi --hmc --hmc_T 100 --hmc_L 100 --hmc_eps 0.01 --rwmh --rwmh_T 100 --verbose



# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 5 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --hmc --rwmh

$SHELL
