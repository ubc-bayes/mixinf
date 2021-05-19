#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10 --reps 1 2 3 4 5 6 7 8 9 10 --tol 0.1 0.05 0.01 --stop 'median' --lbvi -N 5 --kernel 'gaussian' --rkhs 'rbf' --maxiter 5 -t 5 -T 5 --weight_max 20 --ubvi --ubvi_kernels 10 --ubvi_init 100 --ubvi_inflation 5 --ubvi_logfg 100 --ubvi_adamiter 100 --bvi --bvi_kernels 5 --bvi_diagonal --bvi_init 100 --bvi_alpha 100 --gvi --gvi_iter 100 --hmc --hmc_T 50 --hmc_L 100 --hmc_eps 0.1 --rwmh --rwmh_T 50 --cleanup --verbose
python3 -W ignore ../tests.py -d 1 --target '4-mixture' -B 10000 --reps 1 --tol 0.0 --stop 'median' --lbvi -N 5 --kernel 'gaussian' --rkhs 'rbf' --maxiter 10 -t 25 -T 50 --weight_max 20 --cleanup --verbose
#--ubvi --ubvi_kernels 10 --ubvi_init 100 --ubvi_inflation 5 --ubvi_logfg 100 --ubvi_adamiter 100 --bvi --bvi_kernels 5 --bvi_diagonal --bvi_init 100 --bvi_alpha 100 --gvi --gvi_iter 100 --hmc --hmc_T 50 --hmc_L 100 --hmc_eps 0.1 --rwmh --rwmh_T 50



# plot
#python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 10 --tol 0.1 0.05 0.01 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --rwmh

$SHELL
