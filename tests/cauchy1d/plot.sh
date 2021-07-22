#!/bin/bash

python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 30 --tol 0.0 --lbvi_smc --smc 'smc' --smc_eps 0.05 --smc_sd 1 --smc_T 50 --ubvi --bvi

$SHELL
