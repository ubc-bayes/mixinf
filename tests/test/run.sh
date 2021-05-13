#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target '4-mixture' -B 1000 --reps 1 --tol 0.0 --lbvi -N 10 --kernel 'gaussian' --rkhs 'rbf' --maxiter 20 -t 25 -T 25 --weight_max 20 --cleanup --verbose
#--ubvi --ubvi_kernels 20 --ubvi_init 5000 --ubvi_inflation 1 --ubvi_logfg 5000 --ubvi_adamiter 5000 --bvi --bvi_kernels 20 --bvi_diagonal --gvi --hmc --hmc_T 5000 --hmc_L 100 --hmc_eps 0.1 --rwmh --rwmh_T 5000 --cleanup --verbose



# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target '4-mixture' --reps 1 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf'
# --ubvi --bvi --gvi --rwmh

$SHELL
