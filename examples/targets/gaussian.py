# returns std gaussian k-dim density and sampler
import numpy as np


# CREATE DENSITY ####
def p_aux(x, K):
    return np.squeeze((-0.5 * x**2).sum(axis = -1))


# CREATE SAMPLER ####
def sample(size, K): return 3 * np.random.randn(size, K)
