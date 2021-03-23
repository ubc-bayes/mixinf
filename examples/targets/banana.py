# returns banana K-dim density and sampler
import numpy as np
from scipy.special import gamma
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

# see http://probability.ca/jeff/ftpdir/adaptex.pdf



# CREATE DENSITY ####
def p_aux(x, K, b = 0.1):
    banana = tfp.experimental.inference_gym.targets.Banana(ndims=K, curvature=b)
    return np.array(np.exp(banana.unnormalized_log_prob(x)))


# CREATE SAMPLER ####
def sample(size, K, noise = 2, b = 0.1):
    banana = tfp.experimental.inference_gym.targets.Banana(ndims=K, curvature=b)
    return np.array(banana.sample(sample_shape=(size)) + noise*np.random.rand(size, K))
