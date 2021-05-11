#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10 --reps 6 7 8 9 10 --tol 0.0 --stop 'median' --lbvi -N 25 --kernel 'gaussian' --rkhs 'rbf' --maxiter 2 -t 1 -T 500 --weight_max 20 --ubvi_kernels 2 --ubvi --ubvi_init 100 --ubvi_inflation 16 --ubvi_logfg 100 --ubvi_adamiter 100 --bvi --bvi_kernels 2 --bvi_diagonal --gvi --hmc --hmc_T 100 --hmc_L 100 --hmc_eps 0.01 --rwmh --rwmh_T 100 --verbose



# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 5 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --hmc --rwmh

$SHELL
