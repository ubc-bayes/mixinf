# returns cauchy k-dim density and sampler
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad
import scipy.stats as stats
from scipy.special import loggamma


# CREATE DENSITY ####

gamma = 1
mu = 0

#def logp_aux(x, K = 1): return -np.log(np.pi * gamma) - np.log(1 + np.sum((x - mu)**2, axis = -1) / gamma)
def logp_aux(x, K = 1):
    # x is shape(N,K)
    mu =  np.zeros(K)    # location parameter, defaults to 0
    nu = 0.5 * (1+K)
    return np.squeeze(loggamma(nu) - nu*np.log(np.pi) -  nu*np.log( 1 + ((x - mu)**2).sum(axis = -1) ))


# CREATE SAMPLER ####
def sample(size, K): return 4 * np.random.randn(size, K)



# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    if k == 0: return 1000
    if long_opt: return 100
    return 100

def w_schedule(k):
    if k == 0: return 1
    return 0.01
