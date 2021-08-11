#!/bin/bash

# 2-d double banana simulation
python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 10000 --reps 1 --tol 0.0 --stop 'default' --kl --plot --lbvi_smc -N 25 --smc 'smc' --maxiter 20 --smc_eps 0.05 --smc_sd 1. --smc_T 25 --smc_w_maxiter 20 --smc_b_maxiter 10 --smc_cacheing --verbose --seed 123

#python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'default' --kl --plot -N 50 --ubvi --ubvi_kernels 20 --ubvi --ubvi_init 10000 --ubvi_inflation 1 --ubvi_logfg 10000 --ubvi_adamiter 10000 --verbose --seed 123

#python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'default' --kl -N 50 --bvi --bvi_kernels 20 --bvi_diagonal --verbose --seed 123

$SHELL
