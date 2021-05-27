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
#import pickle as pk
import warnings
from timeit import default_timer as timer

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../lbvi/'))
import lbvi


# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="run lbvi other vi and mcmc methods for comparison")

parser.add_argument('-d', '--dim', type = int,
help = 'dimension on which to run both optimizations')
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy', '5-mixture', 'banana', 'double-banana', 'banana-gaussian', 'four-banana', 'double-banana-gaussian', 'network'],
help = 'target distribution to use')
parser.add_argument('-B', type = int, default = 500,
help = 'MC sample size for gradient estimation in SGD')
parser.add_argument('--reps', type = int, default = 1, nargs = '+',
help = 'sequence with repetition numbers - affects name of files being saved, so a sequence is needed')
parser.add_argument('--tol', type = float, nargs = '+',
help = 'sequence of step size tolerances at which to stop alg if maxiter not exceeded')
parser.add_argument('--stop', type = str, default = 'default', choices=['default', 'median'],
help = 'stopping criterion for lbvi, bvi, and ubvi. Default is ksd tolerance for lbvi and iter number for the other. Median is using custom ksd with median sq distance bw')
parser.add_argument('--kl', action = "store_true",
help = 'if specified, kl is calculated for boosting methods and stored; else, ksd is calculated and stored')
parser.add_argument('--lbvi', action = "store_true",
help = 'run lbvi?')
parser.add_argument('-N', type = int,
help = 'sample size on which to run lbvi optimization')
parser.add_argument('--kernel', type = str, default = 'gaussian', choices=['gaussian', 'network'],
help = 'kernel to use in lbvi mixtures')
parser.add_argument('--rkhs', type = str, default = 'rbf', choices=['rbf'],
help = 'RKHS kernel to use for lbvi')
parser.add_argument('--gamma', type = float, default = 1.,
help = 'if rbf kernel is used, the kernel bandwidth')
parser.add_argument('--maxiter', type = int, default = 10,
help = 'maximum number of lbvi iterations')
parser.add_argument('-t', '--t_inc', type = int, default = 25,
help = 'step size increment for chain running in lbvi')
parser.add_argument('-T', '--t_max', type = int, default = 1000,
help = 'maximum number of step sizes allowed per chain in lbvi')
parser.add_argument('--weight_max', type = int, default = 20,
help = 'number of steps before optimizing weights again in lbvi')
parser.add_argument('--no_cache', action = "store_true",
help = 'if specified, lbvi will not cache the mcmc samples')
parser.add_argument('--lbvi_recycle', action = "store_true",
help = 'if specified, lbvi will recycle the weight optimization sample')
parser.add_argument('--ubvi', action = "store_true",
help = 'run ubvi?')
parser.add_argument('--ubvi_kernels', type = int, default = 20,
help = 'number of kernels to add in the ubvi mixture')
parser.add_argument('--ubvi_init', type = int, default = 10000,
help = 'number of iterations for component initialization in ubvi')
parser.add_argument('--ubvi_inflation', type = float, default = 16,
help = 'inflation for new component initialization in ubvi')
parser.add_argument('--ubvi_logfg', type = int, default = 10000,
help = 'number of iterations for logfg estimation in ubvi')
parser.add_argument('--ubvi_adamiter', type = int, default = 10000,
help = 'number of iterations for adam weight optimization in ubvi')
parser.add_argument('--bvi', action = "store_true",
help = 'run bbbvi?')
parser.add_argument('--bvi_diagonal', action = "store_true",
help = 'should bbbvi be run with a diagonal covariane matrix?')
parser.add_argument('--bvi_kernels', type = int, default = 20,
help = 'number of kernels to add in the bvi mixture')
parser.add_argument('--bvi_init', type = int, default = 1000,
help = 'number of iterations to find the new mixture')
parser.add_argument('--bvi_alpha', type = int, default = 1000,
help = 'number of iterations to optimize the weights')
parser.add_argument('--gvi', action = "store_true",
help = 'run standard gaussian vi?')
parser.add_argument('--gvi_iter', type = int, default = 5000,
help = 'number of iterations to find GVI approximation')
parser.add_argument('--hmc', action = "store_true",
help = 'run hamiltonian monte carlo?')
parser.add_argument('--hmc_T', type = int, default = 1000,
help = 'number of steps to run hmc for')
parser.add_argument('--hmc_L', type = int, default = 100,
help = 'number of steps to run leapfrog within hmc')
parser.add_argument('--hmc_eps', type = float, default = 0.01,
help = 'step size for leapfrog within hmc')
parser.add_argument('--rwmh', action = "store_true",
help = 'run random-walk metropolis-hastings?')
parser.add_argument('--rwmh_T', type = int, default = 1000,
help = 'number of steps to run rwmh for')
parser.add_argument('--outpath', type = str, default = '',
help = 'path of file to output')
parser.add_argument('--seed', type = int, default = -1,
help = 'seed for reproducibility')
parser.add_argument('-v', '--verbose', action = "store_true",
help = 'should updates on the stats of the algorithm be printed?')
parser.add_argument('--cleanup', action = "store_true",
help = 'should files from previous experiments be deleted?')

args = parser.parse_args()




# FOLDER SETTINGS ###############
# retrieve flags and folder settings
lbvi_flag = args.lbvi
ubvi_flag = args.ubvi
bvi_flag = args.bvi
bvi_diagonal = args.bvi_diagonal
gvi_flag = args.gvi
rwmh_flag = args.rwmh
hmc_flag = args.hmc
path = args.outpath
cleanup = args.cleanup


if not os.path.exists(path + 'results/'):
    os.makedirs(path + 'results/')

    # create all plotting directories, even if we won't be running smth
    os.makedirs(path + 'results/lbvi/')
    os.makedirs(path + 'results/lbvi/plots/')
    os.makedirs(path + 'results/lbvi/plots/weight_trace/')
    os.makedirs(path + 'results/ubvi/')
    os.makedirs(path + 'results/bvi/')
    os.makedirs(path + 'results/bvi/plots/')
    os.makedirs(path + 'results/gvi/')
    os.makedirs(path + 'results/rwmh/')
    os.makedirs(path + 'results/hmc/')
else:
    # if you have to rerun X and want to cleanup, delete and recreate its directory
    if lbvi_flag and cleanup:
        shutil.rmtree(path + 'results/lbvi/')
        os.makedirs(path + 'results/lbvi/')
        os.makedirs(path + 'results/lbvi/plots/')
        os.makedirs(path + 'results/lbvi/plots/weight_trace/')
    if ubvi_flag and cleanup:
        shutil.rmtree(path + 'results/ubvi/')
        os.makedirs(path + 'results/ubvi/')
        #open('results/ubvi/results.pk', 'a').close()
    if bvi_flag and cleanup:
        shutil.rmtree(path + 'results/bvi/')
        os.makedirs(path + 'results/bvi/')
        os.makedirs(path + 'results/bvi/plots/')
    if gvi_flag and cleanup:
        shutil.rmtree(path + 'results/gvi/')
        os.makedirs(path + 'results/gvi/')
    if hmc_flag and cleanup:
        shutil.rmtree(path + 'results/hmc/')
        os.makedirs(path + 'results/hmc/')
    if rwmh_flag and cleanup:
        shutil.rmtree(path + 'results/rwmh/')
        os.makedirs(path + 'results/rwmh/')

# finally, rename path and create times directorybvi
path = path + 'results/'
if not os.path.exists(path + 'aux/'): os.makedirs(path + 'aux/')
if not os.path.exists(path + 'settings'): os.makedirs(path + 'settings')
###########################




# SETTINGS ####

# simulation settings
K = args.dim
N = args.N
extension = 'pdf'
verbose = args.verbose
B = args.B
weight_max = args.weight_max
reps = np.array(args.reps)
no_reps = reps.shape[0]
seed0 = args.seed
stop = args.stop
tols = np.array(args.tol)
no_tols = tols.shape[0]
klcalc = args.kl

# ALGS SETTINGS
# lbvi
maxiter = args.maxiter
t_increment = args.t_inc
t_max = args.t_max
cacheing = not args.no_cache
lbvi_recycle = args.lbvi_recycle
lbvi_gamma = args.gamma
# ubvi
ubvi_kernels = args.ubvi_kernels
ubvi_init = args.ubvi_init
ubvi_inflation = args.ubvi_inflation
ubvi_logfg = args.ubvi_logfg
ubvi_adamiter = args.ubvi_adamiter
# bvi
bvi_kernels = args.bvi_kernels
bvi_init = args.bvi_init
bvi_alpha = args.bvi_alpha
# gvi
gvi_iter = args.gvi_iter
# hmc
hmc_T = args.hmc_T
hmc_L = args.hmc_L
hmc_eps = args.hmc_eps
# rwmh
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

if target == 'four-banana':
    from targets.four_banana import *
    plt_lims = np.array([-3, 3, -3, 3])

if target == 'double-banana-gaussian':
    from targets.doublebanana_gaussian import *
    plt_lims = np.array([-25, 25, -30, 30])

if target == 'network':
    from targets.networkpdf import logp_aux, sample, w_maxiters, w_schedule
    plt_lims = None



# import kernel for mixture
sample_kernel = args.kernel
if sample_kernel == 'gaussian':
    from kernels.gaussian import *
if sample_kernel == 'network':
    from kernels.network import kernel_sampler

# if running lbvi, init auxiliary arrays
if lbvi_flag:
    lbvi_times = np.zeros((no_reps, no_tols))
    lbvi_ksd = np.zeros((no_reps, no_tols))
    lbvi_kernel = np.zeros((no_reps, no_tols))

# if running ubvi, import suite of functions and init aux arrays
if ubvi_flag:
    import ubvi
    ubvi_times = np.zeros((no_reps, no_tols))
    ubvi_ksd = np.zeros((no_reps, no_tols))
    ubvi_kernel = np.zeros((no_reps, no_tols))

# if running bvi, import suite of functions
if bvi_flag:
    import bvi
    bvi_times = np.zeros((no_reps, no_tols))
    bvi_ksd = np.zeros((no_reps, no_tols))
    bvi_kernel = np.zeros((no_reps, no_tols))

# if running standard gaussian vi, import function
if gvi_flag:
    from bvi import new_gaussian as gvi

# if running hmc, import code for that
if hmc_flag: import hmc

# if running rwmh, import gaussian rwmh kernel
if rwmh_flag:
    from kernels.gaussian import r_sampler as rwmh_initial_sampler
    from kernels.gaussian import gaussian_sampler as rwmh_sampler


# import RKHS kernel
rkhs = args.rkhs
if rkhs == 'rbf':
    from RKHSkernels.rbf import *
kernel, dk_x, dk_y, dk_xy = get_kernel(lbvi_gamma)

# SIMULATION ####
if verbose: print('LBVI comparison')
if verbose: print()
if verbose: print('approximating a ' + target + ' distribution of dimension ' + str(K))
if verbose:
    if no_reps == 1:
        print('running a single comparison')
    else:
        print('running ' + str(no_reps) + ' comparisons')

# print which algorithms are being run

if lbvi_flag: print('running LBVI')
if ubvi_flag: print('running UBVI')
if bvi_flag: print('running BVI')
if gvi_flag: print('running Gaussian VI')
if hmc_flag: print('running Hamiltonian Monte Carlo')
if rwmh_flag: print('running RWMH')
if verbose: print()

r_counter = 1
for r in reps:
    print('simulation ' + str(r_counter) + '/' + str(no_reps))
    for i in range(no_tols):
        tol = tols[i]
        if verbose: print('tolerance: ' + str(tol))


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

        # this just appends all the settings of the methods that are being run into a text file, then saves the file for reproducibility
        file_name = path + 'settings/settings_iter-' + str(r) + '_tol-' + str(tol) + '.txt'
        if not os.path.isfile(file_name):
            # if file does not exist, initialize with info
            settings_text = 'lbvi comparison settings\n\ntarget: ' + target + '\ndimension: ' + str(K) + '\ngradient MC sample size: ' + str(B) + '\nstopping criterion: ' + stop + '\ntolerance: ' +     str(tol) + '\nrandom seed: ' + str(seed)
        else:
            # if file exists, no need to initilaize
            settings_text = ''

        # depending on which methods are being run, modify what is being appended to file
        if lbvi_flag:
            settings_text += '\n\nlbvi settings:' + '\ninitial sample size: ' + str(N) + '\nkernel sampler: ' + sample_kernel + '\nrkhs kernel: ' +    rkhs + '\nstep increments: ' + str(t_increment) + '\nmax no. of steps per kernel: ' + str(t_max) + '\nmax no. of steps before optimizing weights again: ' +     str(weight_max) + '\nmax no of iterations: ' + str(maxiter)
        if ubvi_flag:
            settings_text += '\n\nubvi settings:' + '\nno. of kernels to add: ' + str(ubvi_kernels) + '\ncomponent initialization sample size: ' + str(ubvi_init) + '\ncomponent initialization inflation: ' + str(ubvi_inflation) + '\nlogfg estimation sample size: ' + str(ubvi_logfg) + '\nadam weight optimization iterations: ' + str(ubvi_adamiter)
        if bvi_flag:
            settings_text +=  '\n\nbvi settings:' + '\nno. of kernels to add: ' + str(bvi_kernels)
            if bvi_diagonal:
                settings_text +=  '\ndiagonal covariance matrix'
            else:
                settings_text +=  '\nfull covariance matrix'
        if hmc_flag:
            settings_text +=  '\n\nhmc settings:' + '\nno. of steps to run chain for: ' + str(hmc_T) + '\nno. of steps to run leapfrog integrator for: ' + str(hmc_L) + '\nstep size of leapfrog integrator: ' + str(hmc_eps)
        if rwmh_flag:
            settings_text +=  '\n\nrwmh settings:' + '\nno. of steps to run chain for: ' + str(rwmh_T)

        # finally, open the file and append current settings
        # os_RDWR gives reading and writing permissions; os.O_APPEND tells python to append instead of rewriting the file; O_CREAT tells python to create the file if it doesn't exist already
        settings = os.open(file_name, os.O_RDWR|os.O_APPEND|os.O_CREAT)
        os.write(settings, settings_text.encode())
        os.close(settings)


        # define target log density
        def logp(x): return logp_aux(x, K)

        # score and up function for ksd estimation
        sp = egrad(logp) # returns (N,K)
        up = lbvi.up_gen(kernel, sp, dk_x, dk_y, dk_xy)
        stop_up = None
        if klcalc:
            kl_psample = lambda size : p_sample(size,K)
        else:
            kl_psample = None

        # if running median stopping criterion, import functions and create ksd
        if stop == 'median':
            import RKHSkernels.rbf_median as rbfm
            p_x = p_sample(1000, K)
            p_k, p_kdx, p_kdy, p_kdxdy = rbfm.get_kernel(p_x)   # get kernel and derivatives
            stop_up = lbvi.up_gen(p_k, sp, p_kdx, p_kdy, p_kdxdy)  # get aux up function

        if verbose:
            print('stopping criterion: ' + stop)
        if stop == 'median':
            if verbose: print('median squared distance bandwidth: ' + str(rbfm.get_gamma(p_x)))
        if verbose: print()

        if lbvi_flag:
            if verbose: print('LBVI simulation')
            tmp_path = path + 'lbvi/'
            if verbose:
                print('using ' + str(sample_kernel) + ' mcmc sampler')
                print('using ' + str(rkhs) + ' rkhs kernel with bandwidth ' + str(lbvi_gamma))
                print('initial sample size: ' + str(N))


            # generate sample
            if verbose: print('generating sample')
            y = np.unique(sample(N, K), axis=0)


            # run algorithm
            if verbose:
                print('starting lbvi optimization')
                print()
            w, T, obj, cput, act_k, kls = lbvi.lbvi(y, logp, t_increment, t_max, up, kernel_sampler,  w_maxiters = w_maxiters, w_schedule = w_schedule, B = B, maxiter = maxiter, tol = tol, stop_up = stop_up, weight_max = weight_max, cacheing = cacheing, result_cacheing = True, sample_recycling = lbvi_recycle, verbose = verbose, plot = False, gif = False, plt_lims = plt_lims, plot_path = tmp_path + 'plots/', trace = True, p_sample = kl_psample)

            # save results
            if verbose: print('saving lbvi results')
            np.save(tmp_path + 'y_' + str(r) + '_' + str(tol) + '.npy', y)
            np.save(tmp_path + 'w_' + str(r) + '_' + str(tol) + '.npy', w)
            np.save(tmp_path + 'T_' + str(r) + '_' + str(tol) + '.npy', T)
            np.save(tmp_path + 'cput_' + str(r) + '_' + str(tol) + '.npy', cput)
            np.save(tmp_path + 'obj_' + str(r) + '_' + str(tol) + '.npy', obj)
            np.save(tmp_path + 'kernels_' + str(r) + '_' + str(tol) + '.npy', act_k)
            np.save(tmp_path + 'kl_' + str(r) + '_' + str(tol) + '.npy', kls)

            lbvi_times[r_counter-1,i] = cput[-1]
            lbvi_ksd[r_counter-1,i] = obj[-1]
            lbvi_kernel[r_counter-1,i] = act_k[-1]

            # plot trace
            pltobj = obj
            if klcalc: pltobj = obj

            if verbose: print('plotting lbvi objective trace')
            plt.clf()
            plt.plot(1 + np.arange(pltobj.shape[0]), pltobj, '-k')
            plt.xlabel('iteration')
            plt.ylabel('kernelized stein discrepancy')
            if klcalc: plt.ylabel('KL')
            plt.title('objective trace plot')
            plt.savefig(tmp_path + 'lbvi_trace' + str(r) + '_' + str(tol) + '.png', dpi=900)

            if verbose: print('done with LBVI simulation')
            if verbose: print()

        if ubvi_flag:
            if verbose: print('UBVI simulation')
            tmp_path = path + 'ubvi/'

            if verbose:
                print('kernels to add to mixture: ' + str(ubvi_kernels))
                print('iterations for component initialization: ' + str(ubvi_init))
                print('inflation for component initialization: ' + str(ubvi_inflation))
                print('logfg estimation sample size: ' + str(ubvi_logfg))
                print('iterations for weight optimization via adam: ' + str(ubvi_adamiter))


            # creating gaussian dist and adam weight optimizer
            gauss = ubvi.Gaussian(K, True)
            adam = lambda x0, obj, grd : ubvi.ubvi_adam(x0, obj, grd, adam_learning_rate, ubvi_adamiter, callback = gauss.print_perf)


            # run algorithm
            if verbose: print('starting ubvi optimization')
            if verbose: print()
            ubvi_start = timer()
            ubvi_y = ubvi.UBVI(logp, gauss, adam, n_init = ubvi_init, n_samples = B, up = stop_up, n_logfg_samples = ubvi_logfg, tol = tol, init_inflation = ubvi_inflation)
            ubvi_results = []
            for n in range(1,ubvi_kernels+1):
                build = ubvi_y.build(n)
                ubvi_results.append(build)
                print('cpu times: ' + str(build['cput']))
                # assess convergence
                if build['ksd'][-1] < tol:
                    if verbose: print('tolerance reached; breaking')
                    break
            ubvi_end = timer()


            #ubvi_time = np.array([ubvi_end - ubvi_start])
            #np.save(path + 'times/ubvi_time' + str(r) + '_' + str(tol) + '_' + str(seed) + '.npy', ubvi_time)
            if verbose: print()

            # save results ###
            if verbose: print('saving ubvi results')
            no_ubvi_kernels = len(ubvi_results)


            # extract results
            mus = ubvi_results[no_ubvi_kernels-1]['mus']#[:jstar,:]
            Sigs = ubvi_results[no_ubvi_kernels-1]['Sigs']#[:jstar,...]
            weights = ubvi_results[no_ubvi_kernels-1]['weights']#[:jstar]
            weights = weights / weights.sum()
            cput = ubvi_results[no_ubvi_kernels-1]['cput'][1:]
            obj = ubvi_results[no_ubvi_kernels-1]['ksd'][1:]
            act_k = ubvi_results[no_ubvi_kernels-1]['active_kernels'][1:]

            # save them
            np.save(tmp_path + 'means_' + str(r) + '_' + str(tol) + '.npy', mus)
            np.save(tmp_path + 'covariances_' + str(r) + '_' + str(tol) + '.npy', Sigs)
            np.save(tmp_path + 'weights_' + str(r) + '_' + str(tol) + '.npy', weights)
            np.save(tmp_path + 'cput_' + str(r) + '_' + str(tol) + '.npy', cput)
            np.save(tmp_path + 'obj_' + str(r) + '_' + str(tol) + '.npy', obj)
            np.save(tmp_path + 'kernels_' + str(r) + '_' + str(tol) + '.npy', act_k)

            ubvi_times[r_counter-1,i] = cput[-1]
            ubvi_ksd[r_counter-1,i] = obj[-1]
            ubvi_kernel[r_counter-1,i] = act_k[-1]


            # plot trace
            if verbose: print('plotting ubvi ksd trace')
            plt.clf()
            objs = ubvi_results[no_ubvi_kernels-1]['ksd'][1:]
            plt.plot(1 + np.arange(objs.shape[0]), objs, '-k')
            plt.xlabel('iteration')
            plt.ylabel('KSD divergence')
            plt.title('trace plot of KSD')
            plt.savefig(tmp_path + 'ubvi_trace' + str(r) + '_' + str(tol) + '.png', dpi=900)


            if verbose: print('done with UBVI simulation')
            if verbose: print()



        if bvi_flag:
            if verbose: print('BBBVI simulation')
            tmp_path = path + 'bvi/'
            if verbose: print('kernels to add to mixture: ' + str(bvi_kernels))
            if verbose:
                if bvi_diagonal:
                    print('bvi mixture components will have a diagonal covariance matrix')
                else:
                    print('bvi mixture components will have a full covariance matrix')


            if verbose: print('starting bvi optimization')
            if verbose: print()
            # split by whether covariance matrix is full or diagonal
            if bvi_diagonal:
                mus, Sigmas, alphas, objs, cput, act_k, kls = bvi.bvi_diagonal(logp, bvi_kernels, K, regularization, gamma_init, gamma_alpha, maxiter_alpha = bvi_alpha, maxiter_init = bvi_init, B = B, tol = tol, verbose = verbose, traceplot = True, plotpath = tmp_path + 'plots/', stop_up = stop_up)
            else:
                mus, Sigmas, alphas, objs, cput, act_k, kls = bvi.bvi(logp, bvi_kernels, K, regularization, gamma_init, gamma_alpha, maxiter_alpha = bvi_alpha, maxiter_init = bvi_init, B = B, tol = tol, verbose = verbose, traceplot = True, plotpath = tmp_path + 'plots/', stop_up = stop_up)
            if verbose: print()

            # save results
            if verbose: print('saving bvi results')
            np.save(tmp_path + 'means_' + str(r) + '_' + str(tol) + '.npy', mus)
            np.save(tmp_path + 'covariances_' + str(r) + '_' + str(tol) + '.npy', Sigmas)
            np.save(tmp_path + 'weights_' + str(r) + '_' + str(tol) + '.npy', alphas)
            np.save(tmp_path + 'cput_' + str(r) + '_' + str(tol) + '.npy', cput)
            np.save(tmp_path + 'obj_' + str(r) + '_' + str(tol) + '.npy', objs)
            np.save(tmp_path + 'kernels_' + str(r) + '_' + str(tol) + '.npy', act_k)
            np.save(tmp_path + 'kl_' + str(r) + '_' + str(tol) + '.npy', kls)

            bvi_times[r_counter-1,i] = cput[-1]
            bvi_ksd[r_counter-1,i] = objs[-1]
            bvi_kernel[r_counter-1,i] = act_k[-1]


            # plot trace
            pltobj = obj
            if klcalc: pltobj = obj

            if verbose: print('plotting lbvi objective trace')
            plt.clf()
            plt.plot(1 + np.arange(pltobj.shape[0]), pltobj, '-k')
            plt.xlabel('iteration')
            plt.ylabel('kernelized stein discrepancy')
            if klcalc: plt.ylabel('KL')
            plt.title('objective trace plot')
            plt.savefig(tmp_path + 'bvi_trace' + str(r) + '_' + str(tol) + '.png', dpi=900)


            if verbose: print('done with BBBVI simulation')
            if verbose: print()

        if gvi_flag:
            if verbose: print('Gaussian VI simulation')
            tmp_path = path + 'gvi/'
            if verbose: print('using mean and inflated variance from sample of LBVI sampler as starting params')

            if verbose: print('start gaussian vi optimization')
            if verbose: print()
            y = sample(100, K)
            mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = gvi(logp, K, mu0 = np.mean(y, axis=0), var0 = np.amax(np.var(y, axis=0)), gamma_init = gamma_init, B = B, maxiter = gvi_iter, tol = tol, verbose = False, traceplot = True, plotpath = path + 'gvi/', iteration = 1)

            # save results
            if verbose: print('saving gvi results')
            np.save(tmp_path + 'mean_' + str(r) + '_' + str(tol) + '.npy', mu)
            np.save(tmp_path + 'covariance_' + str(r) + '_' + str(tol) + '.npy', Sigma)
            np.save(tmp_path + 'inv_covariance_' + str(r) + '_' + str(tol) + '.npy', SigmaInv)
            np.save(tmp_path + 'logdet_covariance_' + str(r) + '_' + str(tol) + '.npy', SigmaLogDet)

            if verbose: print('done with Gaussian VI simulation')
            if verbose: print()

        if hmc_flag:
            tmp_path = path + 'hmc/'
            if verbose:
                print('HMC simulation')
                print('number of steps to run the chain for: ' + str(hmc_T))
                print('number of steps to run the leapfrog integrator for: ' + str(hmc_L))
                print('integrator step size: ' + str(hmc_eps))
                print()
                print('start running chain')

            p0 = sample(1, K).reshape(K)
            hmc_y = hmc.HMC(logp = logp, sp = sp, K = K, epsilon = hmc_eps, L = hmc_L, T = hmc_T, burnin = 0.25, p0 = p0, verbose = False)

            # save results
            if verbose: print('saving hmc results')
            np.save(tmp_path + 'y_' + str(r) + '_' + str(tol) + '.npy', hmc_y)

            if verbose: print('done with hmc simulation')
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
            np.save(tmp_path + 'y_' + str(r) + '_' + str(tol) + '.npy', y)

            if verbose: print('done with RWMH simulation')
            if verbose: print()

    r_counter += 1


if verbose: print('saving auxiliary arrays')
if lbvi_flag:
    np.save(path + 'aux/lbvi_times.npy', lbvi_times)
    np.save(path + 'aux/lbvi_ksd.npy', lbvi_ksd)
    np.save(path + 'aux/lbvi_kernels.npy', lbvi_kernel)

if ubvi_flag:
    np.save(path + 'aux/ubvi_times.npy', ubvi_times)
    np.save(path + 'aux/ubvi_ksd.npy', ubvi_ksd)
    np.save(path + 'aux/ubvi_kernels.npy', ubvi_kernel)

if bvi_flag:
    np.save(path + 'aux/bvi_times.npy', bvi_times)
    np.save(path + 'aux/bvi_ksd.npy', bvi_ksd)
    np.save(path + 'aux/bvi_kernels.npy', bvi_kernel)


if verbose: print('done with simulation')
if verbose: print()
