# returns banana K-dim density and sampler
import autograd.numpy as np
from scipy.special import gamma
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt
#import tensorflow as tf
#import tensorflow_probability as tfp

# see http://probability.ca/jeff/ftpdir/adaptex.pdf



# CREATE DENSITY ####
#def logp_aux(x, K, b = 0.1):
#    banana = tfp.experimental.inference_gym.targets.Banana(ndims=K, curvature=b)
#    return np.array(banana.unnormalized_log_prob(x))


# CREATE SAMPLER ####
#def sample(size, K, noise = 2, b = 0.1):
#    banana = tfp.experimental.inference_gym.targets.Banana(ndims=K, curvature=b)
#    return np.array(banana.sample(sample_shape=(size)) + noise*np.random.rand(size, K))

def logp_aux(x, K = 2):
    #return -0.5*x[0]^2/100 -0.5*(x[1] + 0.1*x[0]^2 - 100*0.1)^2
    return np.squeeze(-0.5*x[..., 0]**2 / 100 - 0.5*(x[..., 1] + 0.1*x[..., 0]**2 - 100*0.1)**2)

def sample(size, K = 2):
    mu = np.zeros(K)
    sd = 5
    return 5 * np.random.randn(size, K) + mu


# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    if k == 0: return 1000
    if long_opt: return 100
    return 100

def w_schedule(k):
    if k == 0: return 1
    return 0.01
