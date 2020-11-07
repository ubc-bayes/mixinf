# returns k-dim 5-mixture density and sampler
import numpy as np
from scipy.special import gamma
import scipy.stats as stats
import argparse



# AUXILIARY FUNCTIONS FROM NSVMI TO DEAL WITH MIXTURES ####
###########
def norm_logpdf(x, loc = np.array([0]).reshape(1, 1), scale = np.array([1])):
    """
    evaluate isotropic normal logpdf at x with mean loc and sd scale

    - x is an (m+1)d array, where the last dimension accounts for multivariate x
        eg x[0, 0,..., 0, :] is the first observation and has shape K
    - loc is a shape(N, K) array
    - scale is a shape(N,). The covariance matrix is given by scale[i]**2 * np.diag(N) (ie Gaussians are isotropic)

    returns an md array with same shapes as x (except the last dimension)
    """
    K = x.shape[-1]

    return -0.5 * ((x[..., np.newaxis] - loc.T)**2).sum(axis = -2) / scale**2 - 0.5 * K *  np.log(2 * np.pi) - K * np.log(scale)
###########

###########
def mixture_rvs(size, w, x, rho):
    """
    draws a random sample of size size from the mixture defined by w, x, and rho
    x is a shape(N, K) array and w, rho are shape(N,) arrays
    returns a shape(size, K) array
    """
    N = x.shape[0]
    K = x.shape[1]

    inds = np.random.choice(N, size = size, p = w, replace = True) # indices that will be chosen

    #rand = np.random.multivariate_normal(mean = np.zeros(K), cov = np.eye(K), size = size) # sample from standard normal
    rand = np.random.randn(size, K) # sample from standard normal but more efficiently than as above

    # return scaled and translated random draws
    sigmas = rho[inds] # index std deviations for ease
    return rand * sigmas[:, np.newaxis] + x[inds, :]
###########




# CREATE DENSITY ####
def p_aux(x, K):
    # mixture settings
    mix_size = 5
    weights = np.arange(mix_size, 0, -1)**2
    weights = weights / np.sum(weights)
    means = np.repeat(3 * np.arange(mix_size), K).reshape(mix_size, K)
    sd = np.ones(mix_size) / 5

    # evaluate mixture
    ln = norm_logpdf(x, loc = means, scale = sd)
    target = np.log(weights) + ln  # log sum wn exp(ln) = log sum exp(log wn + ln)
    max_value = np.max(target, axis = -1) # max within last axis
    exp_sum = np.exp(target - max_value[..., np.newaxis]).sum(axis = -1)

    return max_value + np.log(exp_sum)
###



# CREATE SAMPLER ####
def sample(size, K):
    # mixture settings
    mix_size = 5
    weights = np.arange(mix_size, 0, -1)**2
    weights = weights / np.sum(weights)
    means = np.repeat(3 * np.arange(mix_size), K).reshape(mix_size, K)
    sd = np.ones(mix_size) / 5

    return mixture_rvs(size, weights, means, sd)
