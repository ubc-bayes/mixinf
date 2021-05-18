# returns 1-dim mixture of 4 gaussians
import autograd.numpy as np


# gaussian log density
def lognorm(x, mu, sd): return -0.5 * np.sum((x - mu)**2, axis = -1) / sd**2 - 0.5*np.log(2*np.pi) - np.log(sd) # gaussian log density

# target mixture settings
mu = np.array([-3, -2, 2, 3])
sd = np.array([0.2, 0.2, 0.2, 0.2])
weights = np.array([0.3, 0.2, 0.3, 0.2])

def LogSumExp(x):
    maxx = np.max(x)
    return maxx + np.log(np.sum(np.exp(x - maxx)))


# create log density (K is unused)
def logp_aux(x, K = 1):
    #out = np.array([])

    #for i in range(x.shape[0]):
    #    exponents = np.log(weights) - 0.5*np.log(2 * np.pi * sd**2) - 0.5*(x[i] - mu)**2 / sd
    #    out = np.append(out, LogSumExp(exponents))

    out = np.zeros([x.shape[0]])

    for i in range(mu.shape[0]):
        out = out + weights[i] * np.exp(lognorm(x, mu[i], sd[i]))

    return np.log(out)



# define sampler
###########
# auxiliary function
def mixture_rvs(size, w, x, rho):
    """
    draws a random sample of size size from the mixture defined by w, x, and rho
    x is a shape(N, K) array and w, rho are shape(N,) arrays
    returns a shape(size, K) array
    """
    N = x.shape[0]
    K = x.shape[1]

    inds = np.random.choice(N, size = size, p = w, replace = True) # indices that will be chosen
    #rand = np.random.randn(size, K) # sample from standard normal but more efficiently than as above
    rand = np.random.randn(size,K) # sample from standard normal but more efficiently than as above
    # return scaled and translated random draws
    sigmas = rho[inds] # index std deviations for ease
    #return rand * sigmas[:, np.newaxis] + x[inds, :]
    return rand * sigmas[:,np.newaxis] + x[inds,:]
###########

def sample(size, K):
    #return np.array([-2.5, 2.5]).reshape(2,1)
    return mixture_rvs(size, weights, np.repeat(mu,K).reshape(mu.shape[0],K), sd * np.ones(weights.shape[0]))


def p_sample(size, K):
    #return np.array([-2.5, 2.5]).reshape(2,1)
    return mixture_rvs(size, weights, np.repeat(mu,K).reshape(mu.shape[0],K), sd * np.ones(weights.shape[0]))


def w_maxiters(k, long_opt = False):
    if k == 0: return 5
    if long_opt: return 5
    return 5

def w_schedule(k):
    #if k == 0: return 1.
    return 5.
