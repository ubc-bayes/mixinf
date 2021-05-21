#!/bin/bash

python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 20 --tol 0.1 0.05 0.01 0.005 0.001 --lbvi --kernel 'gaussian' --rkhs 'rbf' --t_inc 50 --ubvi --bvi --gvi --hmc --rwmh

$SHELL