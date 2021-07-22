#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10000 --reps 1 2 3 --tol 0.0 --stop 'default' -N 20 --kl --lbvi_smc --smc 'smc' --maxiter 5 --smc_wgamma 1. --smc_bgamma 0.5 --smc_eps 0.05 --smc_sd 1 --smc_T 10 --ubvi --ubvi_kernels 5 --ubvi_init 10000 --ubvi_inflation 1 --ubvi_logfg 10000 --ubvi_adamiter 10000 --bvi --bvi_kernels 5 --bvi_diagonal --bvi_init 1000 --bvi_alpha 1000  --cleanup --verbose

python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10000 --reps 1 --tol 0.0 --stop 'default' --kl --lbvi_smc -N 20 --smc 'smc' --maxiter 30 --smc_wgamma 0.1 --smc_bgamma 0.5 --smc_eps 0.05 --smc_sd 1 --smc_T 50 --verbose

# plot
#python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 3 --tol 0.0 --lbvi_smc --smc 'smc' --smc_eps 0.05 --smc_sd 1 --smc_T 10 --ubvi --bvi

$SHELL
