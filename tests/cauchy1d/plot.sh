#!/bin/bash

python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 30 --tol 0.0 --no_dens --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5 --ubvi --bvi --gvi  --rwmh
#--hmc

$SHELL
