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
## auxiliary functions ###
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

def logsumexp(x):
    maxx = np.maximum(x)
    return maxx + np.log(np.sum(np.exp(x-maxx)))




##########################
##########################
## Gaussian functions ####
##########################
##########################
def norm_logpdf(x, mean, sd):
    """
    log PDF of isotropic Normal(loc, sd x I_K)
    Input:
    x    : nd-array, pdf will be evaluated at x; last axis corresponds to dimension and has shape K
    mean : shape(K,) array, mean of the distribution
    sd   : float, isotropic standard deviation
    """
    K = x.shape[-1]
    return -0.5*np.sum((x-mean)**2,axis=-1)/sd**2 -0.5*K*np.log(2*np.pi) - 0.5*K*np.log(sd)

def norm_random(B, mean, sd):
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
### mixture functions  ###
##########################
##########################
def mix_sample(size, logp, w, smc, r_sd, beta, beta_ls):
    """
    Sample from mixture of SMC components
    Input:
    size    : int, size of the sample
    logp    : function, target log density (for SMC)
    w       : (N,) array, contains weights of each component
    smc     : function, generates samples via SMC
    r_sd    : float, std deviation of reference distributions
    beta    : (N,) array, contains betas of each component
    beta_ls : list of arrays, each array contains the discretization of each component

    Output:
    sample  : (size,K) array with sample from mixture
    """

    active = w>0
    tmp_y = y[active,:]
    tmp_w = w[active]
    tmp_beta = beta[active]

    N = tmp_y.shape[0]
    K = tmp_y.shape[1]

    values = np.arange(N)
    counts = np.floor(size*tmp_w).astype(int)
    counts[-1] += size - counts.sum()

    out = np.zeros((1,K))
    for idx in values:
        # for each value, generate a sample of size counts[idx]
        tmp_logr = lambda x : norm_logpdf(x, tmp_y[idx,:], r_sd)
        tmp_r_sample = lambda B : norm_random(B, tmp_y[idx,:], r_sd)
        tmp_beta_ls = beta_ls[idx]
        tmp_beta_ls = tmp_beta_ls[tmp_beta_ls <= tmp_beta[idx]]
        tmp_out,_,_ = smc(logp = logp, logr = tmp_logr, r_sample = tmp_r_sample, B = counts[idx], beta_ls = tmp_beta_ls, Z0 = 1)
        out = np.concatenate((out, tmp_out)) # add to sample
    # end for
    return out[1:,:]

def mix_logpdf(x, logp, w, smc, r_sd, beta, beta_ls):
    """
    Evaluate log pdf of mixture of SMC components
    Input:
    x       : nd-array, point at which to evaluate logpdf; last axis corresponds to dimension
    logp    : function, target log density (for SMC)
    w       : (N,) array, contains weights of each component
    smc     : function, generates samples via SMC
    r_sd    : float, std deviation of reference distributions
    beta    : (N,) array, contains betas of each component
    beta_ls : list of arrays, each array contains the discretization of each component

    Output:
    sample  : (size,K) array with sample from mixture
    """
    # filter out weights == 0
    active = w>0
    tmp_y = y[active,:]
    tmp_w = w[active]
    tmp_beta = beta[active]

    N = tmp_y.shape[0]
    K = tmp_y.shape[1]

    lps = np.zeros(N)
    for n in range(N):
        # for each value, generate estimate normalizing constant
        tmp_logr = lambda x : norm_logpdf(x, tmp_y[idx,:], r_sd)
        tmp_r_sample = lambda B : norm_random(B, tmp_y[idx,:], r_sd)
        tmp_beta_ls = beta_ls[idx]
        tmp_beta_ls = tmp_beta_ls[tmp_beta_ls <= tmp_beta[idx]]

        # run smc and use Z estimate to evaluate logpdf
        _,tmp_Z,_ = smc(logp = logp, logr = tmp_logr, r_sample = tmp_r_sample, B = counts[idx], beta_ls = tmp_beta_ls, Z0 = 1)
        tmp_lp = (1-tmp_beta[idx])*tmp_logr(x) + tmp_beta[idx]*logp(x) # logpdf up to proportionality
        tmp_lp = tmp_w[n]*tmp_lp/tmp_Z # normalize and account for weight
        lps[n] = tmp_lp
    # end for

    return logsumexp(lps)


##########################
##########################
#### weight functions ####
##########################
##########################

## gradient calculation ###
def kl_grad_alpha(alpha, logp, y, w, beta, beta_ls, r_sd, smc, B, n):
    """
    First derivative of KL wrt alpha for component n
    Input: see choose_weight
    """
    if w[n] == 1: raise ValueError('Cant calculate gradient KL if w == 1.')

    beta_ls = [beta_ls[n][beta_ls[n] <= beta[n]] for n in range(N)]

    # generate sample from nth component and define logpdf
    tmp_logr = lambda x : norm_logpdf(x, mean = y[n,:], sd = r_sd)
    tmp_r_sample = lambda B : norm_random(B, mean = y[n,:], sd = r_sd)
    tmp_beta_ls = beta_ls[n]
    theta1,Z1,_ = smc(logp = logp, logr = tmp_logr, r_sample = tmp_r_sample, B = B, beta_ls = tmp_beta_ls, Z0 = 1)
    logqn = lambda x : ((1-beta[n])*tmp_logr(x) + beta[n]*logp(x))/Z1


    # generate sample from mixture minus nth component and define logpdf
    tmp_w = np.copy(w)
    tmp_w[n] = 0.
    tmp_w = tmp_w / (1-w[n]) # normalize
    logq = lambda x : mix_logpdf(x, logp, tmp_w, smc, r_sd, beta, beta_ls)
    theta2 = mix_sample(B, logp, tmp_w, smc, r_sd, beta, beta_ls)

    # define gamma
    def gamma_n(theta):
        exponents = np.array([np.log(w[n]+alpha) + logqn(theta), np.log(1-(alpha/(1-w[n]))) + logq(theta)])
        return logsumexp(exponents) - logp(theta)

    return np.mean(gamma_n(theta1) - gamma_n(theta2)) - (w[n]/(1-w[n]))


def kl_grad2_alpha():
    """
    Second derivative of KL wrt alpha, evaluated at alpha=0
    """

## choosing next component based on weight ###
def choose_weight(logp, y, w, beta, beta_ls, r_sd, smc, B):
    """
    Choose component that results in greatest KL decrease due to weight perturbation
    Input:
    logp    : function, log target density
    y       : (N,K) array with locations
    w       : (N,) array, weights of components
    beta    : (N,) array, betas of components
    beta_ls : list of np arrays, contains discretizations of components
    r_sd    : float, std deviation of reference distributions
    smc     : function, generates samples via SMC
    B       : integer, number of particles ot use in SMC and to estimate gradients

    Output:
    argmin  : component that minimizes the KL
    disc    : estimate of the KL at the optimal component and alpha
    """

    N = y.shape[0]
    K = y.shape[1]
    beta_ls = [beta_ls[n][beta_ls[n] <= beta[n]] for n in range(N)]

    for n in range(N):
        if w[n] == 1:
            # can't perturb weight if it's the only element in mixture
            logq = lambda x : mix_logpdf(x, logp, w, smc, r_sd, beta, beta_ls)
            sampler = lambda B : mix_sample(B, logp, tmp_w, smc, r_sd, beta, beta_ls)
            kls[n] = kl(logq, logp, sampler, B = B)
        else:
            alpha_star = -kl_grad_alpha()/kl_grad2_alpha()         # minimizes second order approximation to kl
            alpha_star = np.min(1-w[n], np.max(-w[n], alpha_star)) # alpha_star in [-wk, 1-wk]

            # calculate kl estimate
            tmp_w = np.copy(w)
            tmp_w[n] = tmp_w[n] + alpha_star # optimal value
            tmp_logq = lambda x : mix_logpdf(x, logp, tmp_w, smc, r_sd, beta, beta_ls)
            tmp_sampler = lambda B : mix_sample(B, logp, tmp_w, smc, r_sd, beta, beta_ls)
            kls[n] = kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = B)
        # end if
    # end for
    argmin = np.argmin(kls)
    return argmin, kls[argmin]





##########################
##########################
#### MAIN FUNCTION #######
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
        tmp_logq = lambda x : norm_logpdf(x, y[n,:], r_sd)
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
    tmp_logq = lambda x : norm_logpdf(x, y[argmin,:], r_sd)
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

        if verbose: print('Iteration ' + str(iter) + '/' + str(maxiter))


        # calculate weights
        disc,argmin = choose_weight()




    return y, w, obj, cpu_time, active_kernels
