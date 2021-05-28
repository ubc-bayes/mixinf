#!/bin/bash

# run simuluation
python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 500 --reps 1 --tol 0.0 --stop 'median' --kl --lbvi -N 20 --kernel 'gaussian' --rkhs 'rbf' --gamma 10 --maxiter 10 -t 25 -T 1000 --weight_max 20 --cleanup --verbose
#--ubvi --ubvi_kernels 2 --ubvi_init 1000 --ubvi_inflation 1 --ubvi_logfg 1000 --ubvi_adamiter 1000 --bvi --bvi_kernels 2 --bvi_diagonal --bvi_init 100 --bvi_alpha 100



# plot
#python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana-gaussian' --reps 1 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5 --ubvi --bvi

$SHELL
