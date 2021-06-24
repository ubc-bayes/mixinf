#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 1 --tol 0.0 --stop 'median' -N 10 --kl --ubvi --ubvi_kernels 10 --ubvi_init 1000 --ubvi_inflation 16 --ubvi_logfg 10000 --ubvi_adamiter 10000 --cleanup --verbose


# plot
#python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 5 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5
#--ubvi --bvi

$SHELL
