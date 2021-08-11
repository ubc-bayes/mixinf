#!/bin/bash

# 1-d cauchy simulation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 1 --tol 0.0 --stop 'default' --kl --lbvi_smc -N 25 --smc 'smc' --maxiter 5 --smc_eps 0.05 --smc_sd 1 --smc_T 100 --smc_w_maxiter 10 --smc_b_maxiter 10 --smc_cacheing --verbose --plot --seed 123

python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 1 --tol 0.0 --lbvi_smc --smc 'smc' --smc_eps 0.05 --smc_sd 1 --smc_T 100

$SHELL
