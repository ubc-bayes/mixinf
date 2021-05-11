#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'median' --lbvi -N 100 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 50 -T 500 --weight_max 20 --ubvi_kernels 50 --ubvi --ubvi_init 5000 --ubvi_inflation 1 --ubvi_logfg 5000 --ubvi_adamiter 500 --bvi_kernels 50 --bvi_diagonal --hmc_T 5000 --hmc_L 100 --hmc_eps 0.01 --rwmh_T 5000 --verbose

# plot
#python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'banana-gaussian' --lbvi --kernel 'gaussian' --rkhs 'rbf' --bvi --gvi --hmc --rwmh

$SHELL
