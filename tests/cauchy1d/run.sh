#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 10 --lbvi -N 50 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 50 -T 2000 --tol 0.0001 --weight_max 20 --ubvi --ubvi_kernels 30 --ubvi_init 10000 --ubvi_inflation 16 --ubvi_logfg 10000 --ubvi_adamiter 10000 --bvi --bvi_kernels 30 --bvi_diagonal --hmc --hmc_T 10000 --hmc_L 100 --hmc_eps 0.01 --rwmh --rwmh_T 1000 --seed 637230 --verbose

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --hmc --rwmh

$SHELL
