# high-dimensional normal-kernel sequential-opt mixinf
# run simulation with argparse parameters

# PREAMBLE ####
import numpy as np
import pandas as pd
from scipy.special import gamma
import scipy.stats as stats
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import argparse
import sys, os

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../mixinf/'))


# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="run normal-kernel variational mixture inference examples")

parser.add_argument('--opt', type = str, default = 'seq', choices=['seq', 'full'],
help = 'optimization routine to use')
parser.add_argument('--sampling', type = str, default = 'cont', choices=['cont', 'fixed'],
help = 'for seq opt, should weights use fixed or continuous sampling')
parser.add_argument('-d', '--dim', type = int, nargs = '+',
help = 'dimensions on which to run optimization')
parser.add_argument('-N', type = int, nargs = '+',
help = 'sample sizes on which to run optimization')
parser.add_argument('--target', type = str, default = 'cauchy', choices=['cauchy', 'mixture', 'banana'],
help = 'target distribution to use')
parser.add_argument('--maxiter', type = int, default = 10,
help = 'maximum number of iterations')
parser.add_argument('-B', type = int, default = 500,
help = 'MC sample size for gradient estimation in sgd')
parser.add_argument('--tol', type = float, default = 0.001,
help = 'step size tolerance at which to stop alg if maxiter not exceeded')
parser.add_argument('--sd', type = float, nargs = '+', default = 1,
help = 'standard deviations (in log scale) to use for grid optimization')
parser.add_argument('--outpath', type = str, default = '',
help = 'path of file to output')
parser.add_argument('--trace', action = "store_true",
help = 'whether to generate and save a traceplot of the objective function per iteration')
parser.add_argument('--tracepath', type = str, default = '',
help = 'path to save traceplots if generated')
parser.add_argument('-v', '--verbose', action = "store_true",
help = 'should updates on the stats of the algorithm be printed?')
parser.add_argument('-p', '--profiling', action = "store_true",
help = 'should the code be profiled? If yes, profiling results will be printed')

args = parser.parse_args()




# FOLDER SETTINGS
path = args.outpath
trace = args.trace
tracepath = args.tracepath

# check if necessary folder structure exists, and create it if it doesn't
if not os.path.exists(path + 'results/'):
    os.makedirs(path + 'results/')

if trace & (not os.path.exists(tracepath + 'results/trace/')):
    os.makedirs(tracepath + 'results/trace/')

if trace: tracepath = tracepath + 'results/trace/'

path = path + 'results/'



# SETTINGS ####

# simulation settings
dims = np.array(args.dim)
ss = np.array(args.N)
extension = 'pdf'
verbose = args.verbose
profiling = args.profiling

# alg settings
opt = args.opt
maxiter = args.maxiter
B = args.B
tol = args.tol
sd = np.array(args.sd)
fixed_sampling = args.sampling
if fixed_sampling == 'cont':
    fixed_sampling = False
else:
    fixed_sampling = True

# import target density and sampler
target = args.target
if target == 'cauchy':
    from targets.cauchy import *
if target == 'mixture':
    from targets.mixture import *
if target == 'banana':
    from targets.banana import *


# SIMULATION ####

if opt == 'seq':
    import nsvmi # functions to do normal seq-opt variational mixture inference

    # create and save seed
    seed = np.random.choice(np.arange(1, 1000000))
    np.random.seed(seed)


    # save simulation details
    if verbose: print('Saving simulation settings')
    settings_text = 'dims: ' + ' '.join(dims.astype('str')) + '\nno. of kernel basis functions: ' + ' '.join(ss.astype('str')) + '\noptimization: ' + opt + '\nsampling: ' + args.sampling + '\nstd deviations: ' + ' '.join(sd.astype('str')) + '\ntarget: ' + target + '\nmax no of iterations: ' + str(maxiter) + '\ngradient MC sample size B: ' + str(B) + '\nalg tolerance ' +    str(tol) + '\nrandom seed: ' + str(seed)
    settings = os.open(path + 'settings.txt', os.O_RDWR|os.O_CREAT) # create new text file for writing and reading
    os.write(settings, settings_text.encode())
    os.close(settings)

    if verbose: print(f'Begin simulation! approximating a {target} density using {opt} optimization')
    # start simulation
    for K in dims:

        def p(x): return p_aux(x, K)

        if verbose: print(f"Dimension K = {K}\n")

        for N in ss:
            if verbose: print(f"No. of kernel basis N = {N}\n")

            # generate sample
            if verbose: print('Generating sample')
            x = sample(N, K)

            # run algorithm
            w, y, rho, q, obj = nsvmi.nsvmi_grid(p, x, sd = sd, tol = tol, maxiter = maxiter, B = B, fixed_sampling = fixed_sampling, trace = trace, path = tracepath, verbose = verbose, profiling = profiling)

            # save results
            if verbose: print('Saving results')
            title = 'results' + '_N' + str(N) + '_K' + str(K) + '_' + str(time.time())
            #out = pd.DataFrame({'x': np.squeeze(y), 'w': w, 'rho': rho})
            out = pd.DataFrame(y)
            out['w'] = w
            out['rho'] = rho
            out.to_csv(path + title + '.csv', index = False)

            # end for
        # end for


            print('done with simulation!')
# end if

if opt == 'full':
    import nfvmi # functions to do normal seq-opt variational mixture inference

    # create and save seed
    seed = np.random.choice(np.arange(1, 1000000))
    np.random.seed(seed)


    # save simulation details
    if verbose: print('Saving simulation settings')
    settings_text = 'dims: ' + ' '.join(dims.astype('str')) + '\nno. of kernel basis functions: ' + ' '.join(ss.astype('str')) + '\noptimization: ' + opt + '\nstd deviations: ' + ' '.join(sd.astype('str')) + '\ntarget: ' + target + '\nmax no of iterations: ' + str(maxiter) + '\ngradient MC sample size B: ' + str(B) + '\nalg tolerance ' +    str(tol) + '\nrandom seed: ' + str(seed)
    settings = os.open(path + 'settings.txt', os.O_RDWR|os.O_CREAT) # create new text file for writing and reading
    os.write(settings, settings_text.encode())
    os.close(settings)


    if verbose: print(f'begin simulation! approximating a {target} density using {opt} optimization')
    # start simulation
    for K in dims:

        def p(x): return p_aux(x, K)

        if verbose: print(f"Dimension K = {K}\n")

        for N in ss:
            if verbose: print(f"No. of kernel basis N = {N}\n")

            # generate sample
            if verbose: print('Generating sample')
            x = sample(N, K)

            # run algorithm
            w, y, rho, q, obj = nfvmi.nfvmi_grid(p, x, rho = sd, type = 'kl', tol = tol, maxiter = maxiter, B = B, b = 0.1, trace = trace, path = tracepath, verbose = verbose, profiling = profiling)


            # save results
            if verbose: print('Saving results')
            title = 'results' + '_N' + str(N) + '_K' + str(K) + '_' + str(time.time())
            #out = pd.DataFrame({'x': np.squeeze(y), 'w': w, 'rho': rho})
            out = pd.DataFrame(y)
            out['w'] = w
            out['rho'] = rho
            out.to_csv(path + title + '.csv', index = False)

            # end for
        # end for


            print('done with simulation!')
# end if
