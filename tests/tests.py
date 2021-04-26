# locally-adapted boosting variational inference
# run simulation with argparse parameters

# PREAMBLE ####
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad
import pandas as pd
import scipy.stats as stats
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import argparse
import sys, os, shutil
import warnings
from timeit import default_timer as timer

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../lbvi/'))
import lbvi
import bvi


# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="run lbvi and bbbvi examples for comparison")

parser.add_argument('-d', '--dim', type = int,
help = 'dimension on which to run both optimizations')
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy', '5-mixture', 'banana', 'double-banana', 'banana-gaussian'],
help = 'target distribution to use')
parser.add_argument('-B', type = int, default = 500,
help = 'MC sample size for gradient estimation in SGD')
parser.add_argument('--reps', type = int, default = 1,
help = 'number of times to run each method')
parser.add_argument('--lbvi', action = "store_true",
help = 'run lbvi?')
parser.add_argument('-N', type = int,
help = 'sample size on which to run lbvi optimization')
parser.add_argument('--kernel', type = str, default = 'gaussian', choices=['gaussian'],
help = 'kernel to use in lbvi mixtures')
parser.add_argument('--rkhs', type = str, default = 'rbf', choices=['rbf'],
help = 'RKHS kernel to use for lbvi')
parser.add_argument('--maxiter', type = int, default = 10,
help = 'maximum number of lbvi iterations')
parser.add_argument('-t', '--t_inc', type = int, default = 25,
help = 'step size increment for chain running in lbvi')
parser.add_argument('-T', '--t_max', type = int, default = 1000,
help = 'maximum number of step sizes allowed per chain in lbvi')
parser.add_argument('--tol', type = float, default = 0.001,
help = 'step size tolerance at which to stop alg if maxiter not exceeded')
parser.add_argument('--weight_max', type = int, default = 20,
help = 'number of steps before optimizing weights again in lbvi')
parser.add_argument('--bvi', action = "store_true",
help = 'run bbbvi?')
parser.add_argument('--bvi_kernels', type = int, default = 20,
help = 'number of kernels to add in the bvi mixture')
parser.add_argument('--rwmh', action = "store_true",
help = 'run random-walk metropolis-hastings?')
parser.add_argument('--rwmh_T', type = int, default = 1000,
help = 'number of steps to run rwmh for')
parser.add_argument('--outpath', type = str, default = '',
help = 'path of file to output')
parser.add_argument('--seed', type = int, default = -1,
help = 'seed for reproducibility')
#parser.add_argument('--plots', action = "store_true",
#help = 'whether to generate and save plots during the optimization')
#parser.add_argument('--plotpath', type = str, default = '',
#help = 'path to save traceplots if generated')
parser.add_argument('-v', '--verbose', action = "store_true",
help = 'should updates on the stats of the algorithm be printed?')

args = parser.parse_args()




# FOLDER SETTINGS ###############
# retrieve flags and folder settings
lbvi_flag = args.lbvi
bvi_flag = args.bvi
rwmh_flag = args.rwmh
path = args.outpath

# check if necessary folder structure exists, and create it if it doesn't; if it does, clean it accordying to what is going to be run
#if not os.path.exists(path + 'results/'):
#    os.makedirs(path + 'results/')
#
#    # create all plotting directories, even if we won't be running smth
#    os.makedirs(path + 'results/lbvi/')
#    os.makedirs(path + 'results/bvi/')
#    os.makedirs(path + 'results/bvi/plots/')
#else:
#    if lbvi_flag and bvi_flag:
#        # if running both again, delete everything and create all plotting directories
#        shutil.rmtree(path + 'results/')
#
#        # create all plotting directories
#        os.makedirs(path + 'results/lbvi/')
#        os.makedirs(path + 'results/bvi/')
#        os.makedirs(path + 'results/bvi/plots/')
#    elif lbvi_flag and not bvi_flag:
#        # if running lbvi but not bvi, leave the bvi directory alone, delete lbvi directory, and recreate it
#        shutil.rmtree(path + 'results/lbvi/')
#        os.makedirs(path + 'results/lbvi/')
#    elif not lbvi_flag and bvi_flag:
#        # if running bvi but not lbvi, leave the lbvi directory alone, delete bvi directory, and recreate it with plotting directory
#        shutil.rmtree(path + 'results/bvi/')
#        os.makedirs(path + 'results/bvi/')
#        os.makedirs(path + 'results/bvi/plots/')


if not os.path.exists(path + 'results/'):
    os.makedirs(path + 'results/')

    # create all plotting directories, even if we won't be running smth
    os.makedirs(path + 'results/lbvi/')
    os.makedirs(path + 'results/bvi/')
    os.makedirs(path + 'results/bvi/plots/')
    os.makedirst(path + 'results/rwmh/')
else:
    # if you have to rerun X, delete and recreate its directory
    if lbvi_flag:
        shutil.rmtree(path + 'results/lbvi/')
        os.makedirs(path + 'results/lbvi/')
    if bvi_flag:
        shutil.rmtree(path + 'results/bvi/')
        os.makedirs(path + 'results/bvi/')
        os.makedirs(path + 'results/bvi/plots/')
    if rwmh_flag:
        shutil.rmtree(path + 'results/rwmh/')
        os.makedirs(path + 'results/rwmh/')

# finally, rename path and create times directorybvi
path = path + 'results/'
if not os.path.exists(path + 'times/'): os.makedirs(path + 'times/')
###########################




# SETTINGS ####

# simulation settings
K = args.dim
N = args.N
extension = 'pdf'
verbose = args.verbose
B = args.B
weight_max = args.weight_max
reps = args.reps
seed0 = args.seed

# algs settings
maxiter = args.maxiter
tol = args.tol
t_increment = args.t_inc
t_max = args.t_max
bvi_kernels = args.bvi_kernels
rwmh_T = args.rwmh_T

# import target density and sampler
target = args.target
if target == '4-mixture':
    from targets.fourmixture import *
    plt_lims = np.array([-6, 6, 0, 1.5])

if target == 'cauchy':
    from targets.cauchy import *
    plt_lims = np.array([-15, 15, 0, 0.4])

if target == '5-mixture':
    from targets.fivemixture import *
    plt_lims = np.array([-3, 15, 0, 1.5])

if target == 'banana':
    from targets.banana import *
    plt_lims = np.array([-15, 15, -15, 15])

if target == 'double-banana':
    from targets.double_banana import *
    plt_lims = np.array([-2.5, 2.5, -1, 2])

if target == 'banana-gaussian':
    from targets.banana_gaussian import *
    plt_lims = np.array([-3, 3, -2, 3])


# import kernel for mixture
sample_kernel = args.kernel
if sample_kernel == 'gaussian':
    from kernels.gaussian import *

# if running rwmh, import gaussian rwmh kernel
if rwmh_flag:
    from kernels.gaussian import r_sampler as rwmh_initial_sampler
    from kernels.gaussian import kernel_sampler as rwmh_sampler


# import RKHS kernel
rkhs = args.rkhs
if rkhs == 'rbf':
    from RKHSkernels.rbf import *


# SIMULATION ####
if verbose: print('LBVI comparison')
if verbose: print()
if verbose: print('approximating a ' + target + ' distribution of dimension ' + str(K))
if verbose:
    if reps == 1:
        print('running a single comparison')
    else:
        print('running ' + str(reps) + ' comparisons')

# print which algorithms are being run

if lbvi_flag: print('running LBVI')
if bvi_flag: print('running BVI')
if rwmh_flag: print('running RWMH')
if verbose: print()

for r in range(reps):
    if verbose: print('simulation ' + str(r+1) + '/' + str(reps))

    # create and save seed
    if seed0 == -1:
        # if not specified, generate seed at random
        seed = np.random.choice(np.arange(1, 1000000))
    else:
        # if specified, use it and account for repetitions by adding small increments
        seed = seed0 + r

    np.random.seed(seed)

    # save simulation details
    if verbose: print('saving simulation settings')
    if verbose: print()

    settings_text = 'lbvi and bvi comparison settings\n\ntarget: ' + target + '\ndimension: ' + str(K) + '\ngradient MC sample size: ' + str(B) + '\ntolerance: ' +     str(tol) + '\nrandom seed: ' + str(seed)
    if lbvi_flag: settings_text = settings_text + '\n\nlbvi settings:' + '\ninitial sample size: ' + str(N) + '\nkernel sampler: ' + sample_kernel + '\nrkhs kernel: ' +    rkhs + '\nstep increments: ' + str(t_increment) + '\nmax no. of steps per kernel: ' + str(t_max) + '\nmax no. of steps before optimizing weights again: ' +     str(weight_max) + '\nmax no of iterations: ' + str(maxiter)
    if bvi_flag: settings_text = settings_text + '\n\nbvi settings:' + '\nno. of kernels to add: ' + str(bvi_kernels)
    if rwmh_flag: settings_text = settings_text + '\n\nrwmh settings:' + '\nno. of steps to run chain for: ' + str(rwmh_T)
    settings = os.open(path + 'settings' + str(r+1) + '.txt', os.O_RDWR|os.O_CREAT) # create new text file for writing and reading
    os.write(settings, settings_text.encode())
    os.close(settings)


    # define target log density
    def logp(x): return logp_aux(x, K)

    if lbvi_flag:
        if verbose: print('LBVI simulation')
        tmp_path = path + 'lbvi/'
        if verbose: print('using ' + str(sample_kernel) + ' mcmc sampler')
        if verbose: print('using ' + str(rkhs) + ' rkhs kernel')
        if verbose: print('initial sample size: ' + str(N))

        # score and up function for ksd estimation
        sp = egrad(logp) # returns (N,K)
        up = lbvi.up_gen(kernel, sp, dk_x, dk_y, dk_xy)

        # generate sample
        if verbose: print('generating sample')
        y = sample(N, K)

        # run algorithm
        if verbose: print('start lbvi optimization')
        if verbose: print()
        lbvi_start = timer()
        w, T, obj = lbvi.lbvi(y, logp, t_increment, t_max, up, kernel_sampler,  w_maxiters = w_maxiters, w_schedule = w_schedule, B = B, maxiter = maxiter, tol = tol,      weight_max = weight_max, verbose = verbose, plot = False, plt_lims = plt_lims, plot_path = '', trace = False)
        lbvi_end = timer()
        lbvi_time = np.array([lbvi_end - lbvi_start])
        np.save(path + 'times/lbvi_time' + str(r+1) + '_' + str(seed) + '.npy', lbvi_time)
        if verbose: print()

        # save results
        if verbose: print('saving lbvi results')
        np.save(tmp_path + 'y_' + str(r+1) + '.npy', y)
        np.save(tmp_path + 'w_' + str(r+1) + '.npy', w)
        np.save(tmp_path + 'T_' + str(r+1) + '.npy', T)

        # plot trace
        if verbose: print('plotting lbvi objective trace')
        plt.clf()
        plt.plot(1 + np.arange(obj.shape[0]), obj, '-k')
        plt.xlabel('iteration')
        plt.ylabel('kernelized stein discrepancy')
        plt.title('trace plot of ksd')
        plt.savefig(tmp_path + 'lbvi_trace' + str(r+1) + '.png', dpi=900)

        if verbose: print('done with LBVI simulation')
        if verbose: print()


    if bvi_flag:
        if verbose: print('BBBVI simulation')
        tmp_path = path + 'bvi/'
        if verbose: print('kernels to add to mixture: ' + str(bvi_kernels))

        if verbose: print('start bvi optimization')
        if verbose: print()
        bvi_start = timer()
        mus, Sigmas, alphas, objs = bvi.bvi(logp, bvi_kernels, K, regularization, gamma_init, gamma_alpha, B, verbose = verbose, traceplot = True, plotpath = path + 'bvi/plots/')
        bvi_end = timer()
        bvi_time = np.array([bvi_end - bvi_start])
        np.save(path + 'times/bvi_time' + str(r+1) + '_' + str(seed) + '.npy', bvi_time)
        if verbose: print()

        # save results
        if verbose: print('saving bvi results')
        np.save(tmp_path + 'means_' + str(r+1) + '.npy', mus)
        np.save(tmp_path + 'covariances_' + str(r+1) + '.npy', Sigmas)
        np.save(tmp_path + 'weights_' + str(r+1) + '.npy', alphas)


        # plot trace
        if verbose: print('plotting bvi objective trace')
        plt.clf()
        plt.plot(1 + np.arange(objs.shape[0]), objs, '-k')
        plt.xlabel('iteration')
        plt.ylabel('KL divergence')
        plt.title('trace plot of KL')
        plt.savefig(tmp_path + 'bvi_trace' + str(r+1) + '.png', dpi=900)


        if verbose: print('done with BBBVI simulation')
        if verbose: print()


    if rwmh_flag:
        if verbose: print('RWMH simulation')
        tmp_path = path + 'rwmh/'
        if verbose: print('number of steps to run the chain for: ' + str(rwmh_T))

        if verbose: print('start running chain')
        if verbose: print()
        y0 = rwmh_initial_sampler(1, np.zeros((1,K))) # generate initial sample from proposal distribution
        y = np.squeeze(rwmh_sampler(y0, rwmh_T*np.ones(1), 10000, logp), axis=1) # generate sample of size 10,000 starting at y0

        # save results
        if verbose: print('saving rwmh results')
        np.save(tmp_path + 'y_' + str(r+1) + '.npy', y)

        if verbose: print('done with RWMH simulation')
        if verbose: print()




if verbose: print('done with simulation')
if verbose: print()
