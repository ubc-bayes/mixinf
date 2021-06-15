#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target '4-mixture' -B 1000 --reps 1 --tol 0.0 --stop 'default' --kl --ulbvi  -N 5 --kernel 'gaussian' --rkhs 'rbf' --gamma 10 --maxiter 25 --ut 10 -T 110 --weight_max 20 --no_cache --cleanup --verbose


# plot
#python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 5 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5
#--ubvi --bvi

$SHELL
