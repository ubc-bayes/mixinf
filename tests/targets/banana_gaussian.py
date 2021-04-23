# returns mixture of double banana and four gaussians density and sampler
import autograd.numpy as np
from scipy.special import gamma
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt

# see https://arxiv.org/pdf/1910.12794.pdf and https://arxiv.org/pdf/1806.03085.pdf


def norm_logpdf(x, loc = np.array([0]).reshape(1, 1), scale = np.array([1])):
    K = x.shape[-1]
    return -0.5 * ((x[..., np.newaxis] - loc.T)**2).sum(axis = -2) / scale**2 - 0.5 * K *  np.log(2 * np.pi) - K * np.log(scale)



# mixture settings
weights = 0.25*np.ones(4)
means = np.array([[-2, -1], [-2, 2], [2, -1], [2, 2]])
sd = np.ones(4) / 5

def logp_mixture_aux(x, K):
    # evaluate mixture
    ln = norm_logpdf(x, loc = means, scale = sd)
    target = np.log(weights) + ln  # log sum wn exp(ln) = log sum exp(log wn + ln)
    max_value = np.max(target, axis = -1) # max within last axis
    exp_sum = np.exp(target - max_value[..., np.newaxis]).sum(axis = -1)

    return max_value + np.log(exp_sum)



def logp_banana_aux(x, K = 2):
    sigma1 = 1
    sigma2 = 0.09
    y = np.log(30)
    F = lambda x : np.log((1-x[...,0])**2 + 100*(x[...,1]-x[...,0]**2)**2)
    return np.squeeze(-0.5*np.sum(x**2, axis=-1)/sigma1 - 0.5*(y - F(x))**2/sigma2)


# define mixture
alpha = 0.5
def logp_aux(x, K = 2):
    return np.log(alpha*np.exp(logp_banana_aux(x, K)) + (1-alpha)*np.exp(logp_mixture_aux(x, K)))

def sample(size, K = 2):
    mu = np.zeros(K)
    sd = 2
    return sd * np.random.randn(size, K) + mu


# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    if k == 0: return 100
    if long_opt: return 100
    return 100

def w_schedule(k):
    if k == 0: return 0.1
    return 0.1


# CREATE BVI SCHEDULES
# schedule tuning
b1 = 0.001
gamma_alpha = lambda k : b1 / np.sqrt(k+1)

b2 = 0.001
gamma_init = lambda k : b2 / np.sqrt(k+1)

# regularization
ell = 1
regularization = lambda k : ell / np.sqrt(k+2)
