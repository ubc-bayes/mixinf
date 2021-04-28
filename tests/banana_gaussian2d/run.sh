#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'banana-gaussian' -B 1000 -N 100 --kernel 'gaussian' --rkhs 'rbf' --maxiter 30 -t 25 -T 50 --tol 0.00001 --weight_max 10 --ubvi --ubvi_kernels 15 --ubvi_init 10000 --ubvi_inflation 0.1 --ubvi_logfg 10000 --ubvi_adamiter 10000 --bvi_kernels 50 --hmc_T 10000 --hmc_L 100 --hmc_eps 0.01 --rwmh_T 1000 --seed 696023 --verbose

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'banana-gaussian' --lbvi --kernel 'gaussian' --rkhs 'rbf' --bvi --gvi --hmc --rwmh

$SHELL
