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

# for ubvi code
#def logp_ubvi(x): return (- np.log(1 + x**2) - np.log(np.pi)).flatten()
def logp_ubvi(x): return logp_aux(x,K=1)

# CREATE SAMPLER ####
def sample(size, K): return 15 * np.random.randn(size, K)

# CREATE EXACT SAMPLER FOR AGNOSTIC KSD ###
def p_sample(size, K): return np.random.standard_cauchy((size,K))

# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    #if k > 10: return 10
    #if k == 0: return 100
    #if long_opt: return 50
    return 500


def w_schedule(k):
    #if k == 0: return 0.1
    return 0.005

smc_w_schedule = lambda k : 1./np.sqrt(k)
smc_b_schedule = lambda k : 0.025/np.sqrt(k)

# CREATE UBVI SCHEDULES
adam_learning_rate= lambda itr : 10./np.sqrt(itr+1)
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
