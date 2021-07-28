#!/bin/bash

# 2-d double banana simulation
python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'default' --kl --lbvi_smc -N 20 --smc 'smc' --maxiter 20 --smc_eps 0.05 --smc_sd 1 --smc_T 100 --smc_w_maxiter 10 --smc_b_maxiter 10 --smc_cacheing --verbose

$SHELL
