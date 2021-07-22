#!/bin/bash

# 1-d cauchy simulation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 10000 --reps 1 --tol 0.0 --stop 'default' --kl --lbvi_smc -N 20 --smc 'smc' --maxiter 30 --smc_wgamma 1. --smc_bgamma 0.5 --smc_eps 0.05 --smc_sd 1 --smc_T 100 --smc_w_maxiter 500 --verbose

# 2-d double banana simulation
#python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 10000 --reps 1 --tol 0.0 --stop 'default' --kl --lbvi_smc -N 50 --smc 'smc' --maxiter 20 --smc_wgamma 1. --smc_bgamma 0.5 --smc_eps 0.05 --smc_sd 1 --smc_T 100 --smc_w_maxiter 500 --verbose

$SHELL
