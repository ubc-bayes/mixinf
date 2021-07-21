#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 10000 --reps 1 --tol 0.0 --stop 'default' -N 50 --kl --lbvi_smc --smc 'smc' --maxiter 5 --smc_wgamma 0.00000001 --smc_bgamma 0.5 --smc_eps 0.05 --smc_sd 1 --smc_T 10  --cleanup --verbose


# plot
python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana-gaussian' --reps 1 --tol 0.0 --lbvi_smc --smc 'smc' --smc_eps 0.05 --smc_sd 1 --smc_T 10
#--ubvi --bvi

$SHELL
