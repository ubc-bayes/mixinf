# returns mixture of quadruple banana and four gaussians density and sampler
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
weights = np.ones(4)/4
means = np.array([[0, -2], [-2, 0], [0, 2], [2, 0]])
sd = 0.25*np.ones(4)

def logp_mixture_aux(x, K):
    # evaluate mixture
    ln = norm_logpdf(x, loc = means, scale = sd)
    target = np.log(weights) + ln  # log sum wn exp(ln) = log sum exp(log wn + ln)
    max_value = np.max(target, axis = -1) # max within last axis
    exp_sum = np.exp(target - max_value[..., np.newaxis]).sum(axis = -1)

    return max_value + np.log(exp_sum)



def logp_banana_aux(x, K = 2):
    sigma1 = 0.75
    sigma2 = 1.
    y = np.log(20)
    #theta = np.pi
    #A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) # rotation
    #x = (A@x.T).T
    #x[...,1] = x[...,1] - 1
    F = lambda x : np.log(np.maximum(1e-6,(1-x[...,0])**2 + 100*np.abs((x[...,1]**2-x[...,0]**2)**3)))
    return np.squeeze(-0.5*np.sum(x**2, axis=-1)/sigma1 - 0.5*(y - F(x))**2/sigma2)



alpha = 0.5
def logp_aux(x, K = 2):
    return np.log(alpha*np.exp(logp_banana_aux(x, K)) + (1-alpha)*np.exp(logp_mixture_aux(x, K)))
#def logp(x): return logp_aux(x, 2)

# define ubvi logpdf
def logp_ubvi(x): return logp_aux(x,K=2)

def sample(size, K = 2):
    mu = np.zeros(K)
    sd = 2
    return sd * np.random.randn(size, K) + mu


# CREATE EXACT SAMPLER FOR AGNOSTIC KSD ###
# first define function to sample from gaussian mixture
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
    return rand * np.sqrt(sigmas[:, np.newaxis]) + x[inds]
###########


# now define function to sample from double banana via rwmh
def double_banana_sample(size, burnin = 0.5):
    # Gaussian proposals
    sd = 2
    #def logr(x, y): return -0.5 * x.shape[1] * np.sum((x - y)**2, axis = -1) / sd**2 - 0.5*x.shape[1]*np.log(2*np.pi) - x.shape[1]*np.log(sd) # gaussian log density
    def r_sampler(y): return sd * np.random.randn(1, y.shape[1]) + y # gaussian sampler
    y = np.array([[1, 0]]) # init
    n_steps = int(np.round(size/(1-burnin))) # account for burn-in
    out = np.zeros((n_steps,2))
    for t in range(n_steps):
        tmp_y = r_sampler(y) # generate proposal
        logratio = logp_banana_aux(tmp_y)-logp_banana_aux(y) # log hastings ratio
        if np.log(np.random.rand(1)) < np.minimum(0, logratio):
            out[t,:] = tmp_y # accept proposal
        else:
            out[t,:] = y     # reject proposal
    return out[-size:,:]

# now define p sampler
def p_sample(size, K=2):
    out = np.zeros((size,2)) # init
    inds = np.random.choice([0,1], size = size, p = [alpha, 1 - alpha], replace = True) # select indices from banana/gaussian mixture
    n_banana = inds[inds == 0].shape[0]
    n_gauss = size - n_banana
    out[inds == 0,:] = double_banana_sample(n_banana)        # sample from banana
    out[inds == 1,:] = mixture_rvs(n_gauss, weights, means, sd) # sample from gaussian mixture
    return out


# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    if k > 10: return 10
    if k == 0: return 100
    if long_opt: return 50
    return 20

def w_schedule(k):
    if k == 0: return 1.
    return 0.1


# CREATE UBVI SCHEDULES
adam_learning_rate= lambda itr : 0.5/np.sqrt(itr+1)
ubvi_gamma = lambda itr : 1./np.sqrt(1+itr)


# CREATE BVI SCHEDULES
# schedule tuning
b1 = 0.01
gamma_alpha = lambda k : b1 / np.sqrt(k+1)

b2 = 0.01
gamma_init = lambda k : b2 / np.sqrt(k+1)

# regularization
ell = 1
regularization = lambda k : ell / np.sqrt(k+2)
