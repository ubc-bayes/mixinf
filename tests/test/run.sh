#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'median' --lbvi -N 20 --kernel 'gaussian' --rkhs 'rbf' --maxiter 30 -t 25 -T 50 --weight_max 20 --cleanup --verbose



# plot
#python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 1 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf'

$SHELL
