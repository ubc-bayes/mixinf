#!/bin/bash

python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 30 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5 --ubvi --bvi --gvi --hmc --rwmh

$SHELL
