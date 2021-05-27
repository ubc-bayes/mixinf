#!/bin/bash

python3 ../tests.py --help

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 1000 --reps 1 --tol 0.0 --stop 'median' --kl --lbvi -N 20 --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5 --maxiter 2 -t 100 -T 2100 --weight_max 20 --cleanup --verbose



# plot
#python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana-gaussian' --reps 1 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5 # --ubvi --bvi

$SHELL
