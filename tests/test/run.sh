#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10000 --reps 1 2 3 --tol 0.0 --stop 'default' -N 20 --kl --lbvi_smc --smc 'smc' --maxiter 10 --smc_wgamma 1. --smc_bgamma 0.5 --smc_eps 0.05 --smc_sd 1 --smc_T 10 --ubvi --ubvi_kernels 2 --ubvi_init 10000 --ubvi_inflation 1 --ubvi_logfg 10000 --ubvi_adamiter 10000 --bvi --bvi_kernels 10 --bvi_diagonal --bvi_init 1000 --bvi_alpha 1000  --cleanup --verbose

# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 3 --tol 0.0 --lbvi_smc --smc 'smc' --smc_eps 0.05 --smc_sd 1 --smc_T 10 --ubvi --bvi

$SHELL
