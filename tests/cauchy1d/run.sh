#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 10 --tol 0.01 0.001 --stop 'median' --lbvi -N 50 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 100 -T 1000 --weight_max 20 --ubvi_kernels 10 --ubvi_init 100 --ubvi_inflation 16 --ubvi_logfg 100 --ubvi_adamiter 100 --bvi_kernels 3 --bvi_diagonal --hmc_T 10000 --hmc_L 100 --hmc_eps 0.01 --rwmh_T 1000 --verbose

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 10 --tol 0.01 0.001 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi

$SHELL
