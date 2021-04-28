#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 1 --lbvi -N 25 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 25 -T 500 --tol 0.00001 --weight_max 10 --bvi --bvi_kernels 25 --gvi --rwmh --rwmh_T 1000 --seed 637230 --verbose

python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 1 -N 50 --kernel 'gaussian' --rkhs 'rbf' --maxiter 50 -t 25 -T 1000 --tol 0.00001 --weight_max 10 --bvi_kernels 25 --hmc --hmc_T 10000 --hmc_L 100 --hmc_eps 0.01 --rwmh_T 1000 --seed 637230 --verbose

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --lbvi --kernel 'gaussian' --rkhs 'rbf' --bvi --gvi --hmc --rwmh

$SHELL
