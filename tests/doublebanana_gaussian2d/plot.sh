#!/bin/bash

python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'four-banana' --reps 30 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --hmc --rwmh

$SHELL
