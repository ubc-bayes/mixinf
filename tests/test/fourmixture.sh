#!/bin/bash

# lbvi
python3 -W ignore ../tests.py -d 1 --target '4-mixture' -B 10000 --reps 1 --tol 0.0 --stop 'default' --kl --lbvi_smc -N 10 --smc 'smc' --maxiter 10 --smc_eps 0.05 --smc_sd 1 --smc_T 100 --smc_w_maxiter 20 --smc_b_maxiter 10 --smc_cacheing --verbose --plot

# ubvi
#python3 -W ignore ../tests.py -d 1 --target '4-mixture' -B 10000 --reps 1 --tol 0.0 --stop 'default' --kl -N 10 --smc 'smc' --ubvi --ubvi_kernels 10 --ubvi --ubvi_init 10000 --ubvi_inflation 1 --ubvi_logfg 10000 --ubvi_adamiter 10000 --verbose

# bvi
#python3 -W ignore ../tests.py -d 1 --target '4-mixture' -B 10000 --reps 1 --tol 0.0 --stop 'default' --kl --plot -N 10 --smc 'smc' --bvi --bvi_diagonal --bvi_kernels 10 --verbose

$SHELL
