# locally-adapted boosting variational inference
# run simulation with argparse parameters

# PREAMBLE ####
import numpy as np
import pandas as pd
import scipy.stats as stats
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import argparse
import sys, os
import warnings

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../lbvi/'))


# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="run locally-adapted boosting variational inference examples")

parser.add_argument('-N', type = int, nargs = '+',
help = 'sample sizes on which to run optimization')
parser.add_argument('-d', '--dim', type = int, nargs = '+',
help = 'dimensions on which to run optimization')
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture'],
help = 'target distribution to use')
parser.add_argument('--kernel', type = str, default = 'gaussian', choices=['gaussian'],
help = 'kernel to use in mixtures')
parser.add_argument('--rkhs', type = str, default = 'rbf', choices=['rbf'],
help = 'RKHS kernel to use')
parser.add_argument('--maxiter', type = int, default = 10,
help = 'maximum number of iterations')
parser.add_argument('-t', '--t_inc', type = int, default = 25,
help = 'step size increment for chain running')
parser.add_argument('-T', '--t_max', type = int, default = 1000,
help = 'maximum number of step sizes allowed per chain')
parser.add_argument('-B', type = int, default = 500,
help = 'MC sample size for gradient estimation in SGD')
parser.add_argument('--tol', type = float, default = 0.001,
help = 'step size tolerance at which to stop alg if maxiter not exceeded')
parser.add_argument('--outpath', type = str, default = '',
help = 'path of file to output')
parser.add_argument('--plots', action = "store_true",
help = 'whether to generate and save plots during the optimization')
parser.add_argument('--plotpath', type = str, default = '',
help = 'path to save traceplots if generated')
parser.add_argument('-v', '--verbose', action = "store_true",
help = 'should updates on the stats of the algorithm be printed?')
parser.add_argument('-p', '--profiling', action = "store_true",
help = 'should the code be profiled? If yes, profiling results will be printed')

args = parser.parse_args()




# FOLDER SETTINGS
path = args.outpath
plot = args.plots
plotpath = args.plotpath

# check if necessary folder structure exists, and create it if it doesn't
if not os.path.exists(path + 'results/'):
    os.makedirs(path + 'results/')

if plot & (not os.path.exists(plotpath + 'results/plots/')):
    os.makedirs(plotpath + 'results/plots/')
    if not os.path.exists(plotpath + 'results/plot/weight_trace/'):
        os.makedirs(plotpath + 'results/plots/weight_trace/')

if plot: plotpath = plotpath + 'results/plots/'

path = path + 'results/'



# SETTINGS ####

# simulation settings
dims = np.array(args.dim)
ss = np.array(args.N)
extension = 'pdf'
verbose = args.verbose
profiling = args.profiling
B = args.B

# alg settings
maxiter = args.maxiter
tol = args.tol
t_increment = args.t_inc
t_max = args.t_max

# import target density and sampler
target = args.target
if target == '4-mixture':
    from targets.fourmixture import *

# import kernel for mixture
kernel = args.kernel
if kernel == 'gaussian':
    from kernels.gaussian import *

# import RKHS kernel
rkhs = args.rkhs
if rkhs == 'rbf':
    from RKHSkernels.rbf import *


# SIMULATION ####

import lbvi # functions to do normal seq-opt variational mixture inference


# create and save seed
seed = np.random.choice(np.arange(1, 1000000))
np.random.seed(seed)

# save simulation details
if verbose: print('Saving simulation settings')
settings_text = 'dims: ' + ' '.join(dims.astype('str')) + '\nno. of kernel basis functions: ' + ' '.join(ss.astype('str')) + '\ntarget: ' + target + '\nmax no of terations: ' + str(maxiter) + '\ngradient MC sample size B: ' + str(B) + '\nalg tolerance ' +    str(tol) + '\nrandom seed: ' + str(seed)
settings = os.open(path + 'settings.txt', os.O_RDWR|os.O_CREAT) # create new text file for writing and reading
os.write(settings, settings_text.encode())
os.close(settings)

if verbose: print(f'Begin simulation! approximating a {target} density')
# start simulation
for K in dims:
    # define target log density
    def logp(x): return logp_aux(x, K)
    up = lbvi.up_gen(kernel, sp, dk_x, dk_y, dk_xy)

    if verbose: print(f"Dimension K = {K}\n")

    for N in ss:
        if verbose: print(f"No. of kernel basis N = {N}\n")

        # generate sample
        if verbose: print('Generating sample')
        y = sample(N, K)
        y = np.array([-2.5, 2.5])
        # run algorithm
        w, T, obj = lbvi.lbvi(y, logp, t_increment, t_max, up, kernel_sampler, w_maxiters = None, w_schedule = None, B = B, maxiter = maxiter, tol = tol, verbose = verbose, plot = plot, plot_path = plotpath, trace = plot)

        # save results
        if verbose: print('Saving results')
        title = 'results' + '_N' + str(N) + '_K' + str(K) + '_' + str(time.time())

        out = pd.DataFrame(y)
        out['w'] = w
        out['T'] = T
        out.to_csv(path + title + '.csv', index = False)
        # end for
    # end for

print('done with simulation!')
