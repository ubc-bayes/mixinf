#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 10 --tol 0.01 0.005 0.0025 0.001 0.0005 0.00025 --stop 'median' --lbvi -N 25 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 50 -T 500 --weight_max 20 --ubvi --ubvi_kernels 50 --ubvi_init 5000 --ubvi_inflation 16 --ubvi_logfg 5000 --ubvi_adamiter 5000 --bvi --bvi_kernels 50 --bvi_diagonal --gvi --hmc --hmc_T 5000 --hmc_L 100 --hmc_eps 0.01 --rwmh --rwmh_T 5000 --verbose

python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 10 --tol 0.01 0.005 0.0025 0.001 0.0005 0.00025 --stop 'median' -N 25 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 50 -T 500 --weight_max 20 --ubvi_kernels 50 --ubvi_init 5000 --ubvi_inflation 16 --ubvi_logfg 5000 --ubvi_adamiter 5000 --bvi --bvi_kernels 50 --bvi_diagonal --hmc_T 5000 --hmc_L 100 --hmc_eps 0.01 --rwmh_T 5000 --verbose

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 10 --tol 0.01 0.005 0.0025 0.001 0.0005 0.00025 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --hmc --rwmh

$SHELL
