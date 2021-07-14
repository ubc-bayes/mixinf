# suite of functions for doing locally-adapted boosting variational inference with SMC components

# preamble
import numpy as np
import scipy.stats as stats
import pandas as pd
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import os
import imageio

##########################
##########################
##########################
def kl(logq, logp, sampler, B = 1000, direction = 'reverse'):
    """
    Estimate the KL divergence
    Input:
    logq       : function, log density of variational approximation
    logp       : function, log target density
    sampler    : function, generates samples from either q or p (depending on direction)
    B          : int, number of samples to generate
    direction  : str, either reverse or forward

    Output:
    kl         : float, estimate of KL(q||p) if direction is reverse, and of KL(p||q) if direction is forward
    """
    theta = sampler(B)
    if direction == 'reverse':
        return np.mean(logq(theta) - logp(theta), axis=-1)
    elif direction == 'forward':
        return np.mean(logp(theta) - logq(theta), axis=-1)
    else: raise NotImplementedError
##########################
##########################
##########################

##########################
## Gaussian functions ####
##########################
def norm_pdf(x, loc, sd):
    """
    PDF of isotropic Normal(loc, sd x I_K)
    Input:
    x    : nd-array, pdf will be evaluated at x; last axis corresponds to dimension and has shape K
    mean : shape(K,) array, mean of the distribution
    sd   : float, isotropic standard deviation
    """
    K = x.shape[-1]
    return -0.5*np.sum((x-mean)**2,axis=-1)/sd**2 -0.5*K*np.log(2*np.pi) - 0.5*K*np.log(sd)

def norm_random(B, loc, sd):
    """
    Generate samples from isotropic Normal(loc, sd x I_K)
    Input:
    B    : int, number of samples to draw
    mean : shape(K,) array, mean of the distribution
    sd   : float, isotropic standard deviation
    """
    K = mean.shape[0]
    return sd*np.random.randn(B,K) + mean
##########################
##########################
##########################


##########################
##########################
##########################
def lbvi_smc(y, logp, smc, smc_eps = 0.05, r_sd = None, maxiter = 10, B = 1000, verbose = False):
    """
    Run LBVI with SMC components
    Input:
    y          : (N,K) array, initial sample locations
    logp       : function, target log density
    smc        : function, smc sampler (see readme in tests/smc/)
    smc_eps    : float, step size for smc discretization
    r_sd       : float, std deviation used for reference distributions; if None, 3 will be used
    maxiter    : int, maximum number of iterations to run the main loop for
    B          : int, number of MC samples to use for gradient estimation
    verbose    : boolean, whether to print messages

    Output:
    """
    if verbose:
        print('Running LBVI with SMC components')
        print()

    # init timer and initialize values
    t0 = time.perf_counter()
    N = y.shape[0]
    K = y.shape[1]
    betas = np.zeros(N)
    beta_ls = [np.linspace(0,1,int(1/smc_eps)+1) for n in range(N)]
    w = np.zeros(N)


    ##########################
    ##########################
    # initialize mixture #####
    ##########################
    ##########################
    if verbose: print('Optimizing first component')
    if r_sd is None: r_sd = 3
    # TODO FIND INITIAL SD USING VI PASS
    # is this a good idea? computational cost times N might be large

    tmp_kl = np.zeros(N)
    for n in range(N):
        if verbose: print(str(n+1) + '/' + str(N), end = '\r')
        tmp_sampler = lambda B : norm_random(B, y[n,:], r_sd)
        tmp_logq = lambda x : norm_pdf(x, y[n,:], r_sd)
        tmp_kl[n] = kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = B)
    # end for

    argmin = np.argmin(tmp_kl)    # kl minimizer
    if verbose: print('First component mean: ' + str(y[argmin, 0:min(K,3)]))
    w[argmin] = 1                 # update weight
    active = np.array([argmin])   # init active set


    ##########################
    ##########################
    # estimate objective #####
    ##########################
    ##########################
    obj_timer = time.perf_counter()
    tmp_sampler = lambda B : norm_random(B, y[argmin,:], r_sd)
    tmp_logq = lambda x : norm_pdf(x, y[argmin,:], r_sd)
    obj = np.array([kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = 10000)])
    obj_timer = time.perf_counter() - obj_timer


    ##########################
    ##########################
    # stats print out    #####
    ##########################
    ##########################
    cpu_time = np.array([time.perf_counter() - t0 - obj_timer])
    active_kernels = np.array([1.])
    if verbose:
        print('KL: ' + str(obj[-1]))
        print('CPU time: ' + str(cpu_time[-1]))
        print()


    ##########################
    ##########################
    # start main loop    #####
    ##########################
    ##########################
    for iter in range(1,maxiter+1):

        if verbose: print('Iteration ' + str(iter))


    return y, w, obj, cpu_time, active_kernels
