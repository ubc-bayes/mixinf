#!/bin/bash

# run simulation
#python3 -W ignore ../tests.py -d 2 --target 'banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'median' -N 50 --kernel 'gaussian' --rkhs 'rbf' --maxiter 5 -t 25 -T 500 --weight_max 20 --ubvi_kernels 5 --ubvi_init 5000 --ubvi_inflation 1 --ubvi_logfg 5000 --ubvi_adamiter 500 --bvi_kernels 5 --bvi_diagonal --gvi --hmc --hmc_T 5000 --hmc_L 100 --hmc_eps 0.01 --rwmh --rwmh_T 5000 --cleanup --verbose

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'banana-gaussian' --reps 1 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi  --hmc --rwmh

$SHELL
