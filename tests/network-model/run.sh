#!/bin/bash

# run simulation
python3 -W ignore ../tests.py -d 203 --target 'network' -B 1000 --reps 1 --tol 0.0 --lbvi -N 7 --kernel 'network' --rkhs 'rbf' --maxiter 10 -t 1 -T 1 --weight_max 20 --seed 123 --cleanup --verbose

$SHELL
