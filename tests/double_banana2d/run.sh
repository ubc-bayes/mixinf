#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'double-banana' -B 1000 --reps 1 --lbvi -N 50 --kernel 'gaussian' --rkhs 'rbf' --maxiter 20 -t 25 -T 500 --tol 0.0001 --weight_max 10 --ubvi --ubvi_kernels 30 --ubvi_init 10000 --ubvi_inflation 16 --ubvi_logfg 10000 --ubvi_adamiter 10000 --bvi --bvi_kernels 30 --bvi_diagonal --gvi --hmc --hmc_T 10000 --hmc_L 100 --hmc_eps 0.01 --rwmh --rwmh_T 1000 --seed 637230 --verbose

# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana' --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --hmc --rwmh

$SHELL
