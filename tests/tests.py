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
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy', '5-mixture', 'banana', 'double-banana'],
help = 'target distribution to use')
parser.add_argument('-B', type = int, default = 500,
help = 'MC sample size for gradient estimation in SGD')
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
parser.add_argument('--bvi_kernels', type = int,
help = 'number of kernels to add in the bvi mixture')
parser.add_argument('--outpath', type = str, default = '',
help = 'path of file to output')
#parser.add_argument('--plots', action = "store_true",
#help = 'whether to generate and save plots during the optimization')
#parser.add_argument('--plotpath', type = str, default = '',
#help = 'path to save traceplots if generated')
parser.add_argument('-v', '--verbose', action = "store_true",
help = 'should updates on the stats of the algorithm be printed?')

args = parser.parse_args()




# FOLDER SETTINGS
lbvi_flag = args.lbvi
bvi_flag = args.bvi
path = args.outpath

# check if necessary folder structure exists, and create it if it doesn't; if it does, clean it accordying to what is going to be run
if not os.path.exists(path + 'results/'):
    os.makedirs(path + 'results/')

    # make plotting directories
    os.makedirs(path + 'results/lbvi/')
    os.makedirs(path + 'results/bvi/')
    os.makedirs(path + 'results/bvi/plots/')
else:
    if lbvi_flag and bvi_flag:
        shutil.rmtree(path + 'results/')

        # make plotting directories
        os.makedirs(path + 'results/lbvi/')
        os.makedirs(path + 'results/bvi/')
        os.makedirs(path + 'results/bvi/plots/')
    elif lbvi_flag and not bvi_flag:
        shutil.rmtree(path + 'results/lbvi/')
    elif not lbvi_flag and bvi_flag:
        shutil.rmtree(path + 'results/bvi/')
        os.makedirs(path + 'results/bvi/plots/')

path = path + 'results/'
os.makedirs(path + 'times/')





# SETTINGS ####

# simulation settings
K = args.dim
N = args.N
extension = 'pdf'
verbose = args.verbose
B = args.B
weight_max = args.weight_max

# alg settings
maxiter = args.maxiter
tol = args.tol
t_increment = args.t_inc
t_max = args.t_max
bvi_kernels = args.bvi_kernels

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


# import kernel for mixture
sample_kernel = args.kernel
if sample_kernel == 'gaussian':
    from kernels.gaussian import *


# import RKHS kernel
rkhs = args.rkhs
if rkhs == 'rbf':
    from RKHSkernels.rbf import *


# SIMULATION ####

# create and save seed
seed = np.random.choice(np.arange(1, 1000000))
np.random.seed(seed)

# save simulation details
if verbose: print('saving simulation settings')

settings_text = 'lbvi and bvi comparison settings\n\ntarget: ' + target + '\ndimension: ' + str(K) + '\ngradient MC sample size: ' + str(B) + '\ntolerance: ' +  str(tol) + '\nrandom seed: ' + str(seed)
if lbvi_flag: settings_text = settings_text + '\n\nlbvi settings:' + '\ninitial sample size: ' + str(N) + '\nkernel sampler: ' + sample_kernel + '\nrkhs kernel: ' + rkhs + '\nstep increments: ' + str(t_increment) + '\nmax no. of steps per kernel: ' + str(t_max) + '\nmax no. of steps before optimizing weights again: ' + str(weight_max) + '\nmax no of iterations: ' + str(maxiter)
if bvi_flag: settings_text = settings_text + '\n\nbvi settings:' + '\nno. of kernels to add: ' + str(bvi_kernels)
settings = os.open(path + 'settings.txt', os.O_RDWR|os.O_CREAT) # create new text file for writing and reading
os.write(settings, settings_text.encode())
os.close(settings)


# define target log density
def logp(x): return logp_aux(x, K)


if verbose: print('dimension K = ' + str(K))
if verbose: print()

if lbvi_flag:
    if verbose: print('LBVI simulation')
    tmp_path = path + 'lbvi/'
    if verbose: print('number of kernel basis N = ' + str(N))

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
    w, T, obj = lbvi.lbvi(y, logp, t_increment, t_max, up, kernel_sampler,  w_maxiters = w_maxiters, w_schedule = w_schedule, B = B, maxiter = maxiter, tol = tol,  weight_max = weight_max, verbose = verbose, plot = False, plt_lims = plt_lims, plot_path = '', trace = False)
    lbvi_end = timer()
    lbvi_time = np.array([lbvi_end - lbvi_start])
    np.save(path + 'times/lbvi_time_' + str(seed) + '.npy', lbvi_time)
    if verbose: print()

    # save results
    if verbose: print('saving lbvi results')
    np.save(tmp_path + 'y.npy', y)
    np.save(tmp_path + 'w.npy', w)
    np.save(tmp_path + 'T.npy', T)
    #title = 'lbvi_results' + '_N' + str(N) + '_K' + str(K) + '_' + str(time.time())
    #out = pd.DataFrame(y)
    #out['w'] = w
    #out['steps'] = T
    #ut.to_csv(tmp_path + title + '.csv', index = False)

    # plot trace
    if verbose: print('plotting lbvi objective trace')
    plt.clf()
    plt.plot(1 + np.arange(obj.shape[0]), obj, '-k')
    plt.xlabel('iteration')
    plt.ylabel('kernelized stein discrepancy')
    plt.title('trace plot of ksd')
    plt.savefig(tmp_path + 'lbvi_trace.png', dpi=900)

    if verbose: print('done with LBVI simulation!')
    if verbose: print()


if bvi_flag:
    if verbose: print('BBBVI simulation')
    tmp_path = path + 'bvi/'
    if verbose: print('kernels to add to mixture: ' + str(bvi_kernels))

    if verbose: print('start bvi optimization')
    if verbose: print()
    bvi_start = timer()
    mus, Sigmas, alphas, objs = bvi.bvi(logp, bvi_kernels, K, regularization, gamma_init, gamma_alpha, B, verbose = verbose, traceplot = True, plotpath = path + 'bvi/plots/    ')
    bvi_end = timer()
    bvi_time = np.array([bvi_end - bvi_start])
    np.save(path + 'times/bvi_time_' + str(seed) + '.npy', bvi_time)
    if verbose: print()

    # save results
    if verbose: print('saving bvi results')
    np.save(tmp_path + 'means.npy', mus)
    np.save(tmp_path + 'covariances.npy', Sigmas)
    np.save(tmp_path + 'weights.npy', alphas)


    # plot trace
    if verbose: print('plotting bvi objective trace')
    plt.clf()
    plt.plot(1 + np.arange(objs.shape[0]), objs, '-k')
    plt.xlabel('iteration')
    plt.ylabel('KL divergence')
    plt.title('trace plot of KL')
    plt.savefig(tmp_path + 'bvi_trace.png', dpi=900)


    if verbose: print('done with BBBVI simulation!')
    if verbose: print()


if verbose: print()
