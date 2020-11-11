#!/bin/bash

# run simuluation
python3 -W ignore ../experiment.py --opt 'seq' --sampling 'cont' -d 3 -N 100 --target 'cauchy' --maxiter 20 -B 1000 --tol 0.00001 --sd 1 10 100 --trace --verbose --profiling

$SHELL
