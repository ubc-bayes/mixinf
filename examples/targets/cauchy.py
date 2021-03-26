# returns cauchy k-dim density and sampler
import numpy as np
#from scipy.special import gamma
import scipy.stats as stats
import argparse


# CREATE DENSITY ####


#def log p_aux(x, K):
#    df = 1               # degrees of freedom
#    mu =  0 * np.ones(K) # location parameter, defaults to 0
#    return np.squeeze(- 0.5 * (df + K) * np.log( 1 + ((x - mu)**2).sum(axis = -1) / df ))

gamma = 1
mu = 0

def logp_aux(x, K = 1): return -np.log(np.pi * gamma) - np.log(1 + (x - mu)**2 / gamma)


# CREATE SAMPLER ####
def sample(size, K): return 4 * np.random.randn(size, K)



# CREATE SCORE FUNCTION
def sp(x, K = 1): return -2 * (x - mu) / (gamma + (x - mu)**2)


# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    if k == 0: return 1000
    if long_opt: return 100
    return 100

def w_schedule(k):
    if k == 0: return 0.01
    return 0.0001
