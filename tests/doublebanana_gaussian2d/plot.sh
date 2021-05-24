#!/bin/bash

python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana-gaussian' --reps 50 --tol 0.0 --no_dens --lbvi --kernel 'gaussian' --rkhs 'rbf' --ubvi --bvi --gvi --hmc --rwmh

$SHELL
