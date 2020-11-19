# returns cauchy k-dim density and sampler
import numpy as np
from scipy.special import gamma
import scipy.stats as stats
import argparse


# CREATE DENSITY ####


def p_aux(x, K):
    df = 1               # degrees of freedom
    mu =  0 * np.ones(K) # location parameter, defaults to 0
    return np.squeeze(- 0.5 * (df + K) * np.log( 1 + ((x - mu)**2).sum(axis = -1) / df ))


# CREATE SAMPLER ####
def sample(size, K): return 25 * np.random.randn(size, K)
