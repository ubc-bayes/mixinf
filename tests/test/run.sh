#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 2 --target 'double-banana-gaussian' -B 1000 --reps 1 2 3 4 5 --tol 0.0 --stop 'default' --kl --lbvi -N 20 --kernel 'gaussian' --rkhs 'rbf' --gamma 10 --lbvi_recycle --maxiter 10 -t 25 -T 100 --weight_max 20 --cleanup --verbose
#--ubvi --ubvi_kernels 2 --ubvi_init 1000 --ubvi_inflation 1 --ubvi_logfg 1000 --ubvi_adamiter 1000 --bvi --bvi_kernels 2 --bvi_diagonal --bvi_init 100 --bvi_alpha 100
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 1000 --reps 1 2 3 4 5 --tol 0.0 --stop 'default' --kl --lbvi -N 5 --kernel 'gaussian' --rkhs 'rbf' --gamma 10 --lbvi_recycle --maxiter 20 -t 100 -T 2100 --weight_max 20 --cleanup --verbose


# plot
python3 -W ignore ../univariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'cauchy' --reps 5 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5
#--ubvi --bvi

$SHELL
