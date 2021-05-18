#!/bin/bash

# run simulation
python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'median' -N 25 --kernel 'gaussian' --rkhs 'rbf' --maxiter 20 -t 50 -T 100 --weight_max 20 --ubvi_kernels 20 --ubvi_init 5000 --ubvi_inflation 1 --ubvi_logfg 5000 --ubvi_adamiter 5000 --bvi_kernels 20 --bvi_diagonal --rwmh --rwmh_T 5000 --cleanup --verbose
#  --gvi --hmc --hmc_T 5000 --hmc_L 100 --hmc_eps 0.1 --rwmh --rwmh_T 5000

# plot
#python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana-gaussian' --reps 1 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi  --hmc --rwmh

$SHELL
