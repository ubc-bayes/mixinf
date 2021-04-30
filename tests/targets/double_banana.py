# returns banana K-dim density and sampler
import autograd.numpy as np
from scipy.special import gamma
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt

# see https://arxiv.org/pdf/1910.12794.pdf and https://arxiv.org/pdf/1806.03085.pdf


def logp_aux(x, K = 2):
    sigma1 = 1
    sigma2 = 0.09
    y = np.log(30)
    F = lambda x : np.log((1-x[...,0])**2 + 100*(x[...,1]-x[...,0]**2)**2)
    return np.squeeze(-0.5*np.sum(x**2, axis=-1)/sigma1 - 0.5*(y - F(x))**2/sigma2)

def sample(size, K = 2):
    mu = 0*np.ones(K)
    sd = 0.1
    return sd * np.random.randn(size, K) + mu


# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    if k == 0: return 200
    if long_opt: return 100
    return 100

def w_schedule(k):
    if k == 0: return 0.0001
    return 0.0001


# CREATE UBVI SCHEDULES
adam_learning_rate= lambda itr : 1./np.sqrt(itr+1)
ubvi_gamma = lambda itr : 1./np.sqrt(1+itr)


# CREATE BVI SCHEDULES
# schedule tuning
b1 = 0.01
gamma_alpha = lambda k : b1 / np.sqrt(k+1)

b2 = 0.05
gamma_init = lambda k : b2 / np.sqrt(k+1)

# regularization
ell = 1
regularization = lambda k : ell / np.sqrt(k+2)
