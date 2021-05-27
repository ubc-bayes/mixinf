#!/bin/bash

# run simuluation
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 100 --reps 1 --tol 0.0 --stop 'median' --kl --lbvi -N 5 --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5 --maxiter 20 -t 100 -T 2100 --weight_max 20 --cleanup --verbose
#python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 100 --reps 1 --tol 0.0 --stop 'median' --kl --bvi --bvi_kernels 20 --bvi_diagonal --bvi_init 1000 --bvi_alpha 1000 --cleanup --verbose
python3 -W ignore ../tests.py -d 1 --target 'cauchy' -B 100 --reps 1 --tol 0.0 --stop 'median' --kl --ubvi --ubvi_kernels 20 --ubvi_init 10000 --ubvi_inflation 1 --ubvi_logfg 10000 --ubvi_adamiter 10000 --cleanup --verbose
#--ubvi --ubvi_kernels 20 --ubvi_init 10000 --ubvi_inflation 1 --ubvi_logfg 10000 --ubvi_adamiter 10000 --bvi --bvi_kernels 20 --bvi_diagonal --bvi_init 1000 --bvi_alpha 1000



# plot
#python3 -W ignore ../bivariate-plot.py --inpath 'results/' --outpath 'results/plots/' --target 'double-banana-gaussian' --reps 1 --tol 0.0 --lbvi --kernel 'gaussian' --rkhs 'rbf' --gamma 0.5
#--ubvi --bvi

$SHELL
