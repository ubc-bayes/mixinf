#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10000 --reps 1 --tol 0.0 --stop 'default' -N 10 --kl --lbvi_smc --smc 'smc' --maxiter 10 --smc_wgamma 1. --smc_bgamma 0.1 --smc_eps 0.05 --smc_sd 1 --smc_T 10 --cleanup --verbose


# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 1 --tol 0.0 --lbvi_smc --smc 'smc' --smc_eps 0.05 --smc_sd 1 --smc_T 10
#--ubvi --bvi

$SHELL
