# returns 1-dim mixture of 4 gaussians
import numpy as np


# gaussian log density
def lognorm(x, mu, sd): return -0.5 * (x - mu)**2 / sd - 0.5*np.log(2*np.pi) - np.log(sd) # gaussian log density

# target mixture settings
mu = np.array([-3, -2, 2, 3])
sd = np.array([0.2, 0.2, 0.2, 0.2])
weights = np.array([0.3, 0.2, 0.3, 0.2])

def LogSumExp(x):
    maxx = np.max(x)
    return maxx + np.log(np.sum(np.exp(x - maxx)))


# create log density (K is unused)
def logp_aux(x, K = 1):
    out = np.array([])

    for i in range(x.shape[0]):
        exponents = np.log(weights) - 0.5*np.log(2 * np.pi * sd**2) - 0.5*(x[i] - mu)**2 / sd
        out = np.append(out, LogSumExp(exponents))

    return out



# score function
def sp(x, K = 1):
    out = np.array([])

    for i in range(x.shape[0]):
        exponents = np.log(weights) - 0.5*np.log(2 * np.pi * sd**2) - 0.5*(x[i] - mu)**2 / sd + np.log(np.abs(x[i] - mu)) - 2*np.log(sd)
        out = np.append(out, LogSumExp(exponents) / np.exp(logp_aux(np.array([x[i]]))))

    #out = np.zeros([x.shape[0]])

    #for i in range(mu.shape[0]):
    #    out = out + weights[i] * np.exp(lognorm(x, mu[i], sd[i]) * (x - mu[i])) / sd[i]**2

    return out


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
    #K = x.shape[1]

    inds = np.random.choice(N, size = size, p = w, replace = True) # indices that will be chosen
    #rand = np.random.randn(size, K) # sample from standard normal but more efficiently than as above
    rand = np.random.randn(size) # sample from standard normal but more efficiently than as above
    # return scaled and translated random draws
    sigmas = rho[inds] # index std deviations for ease
    #return rand * sigmas[:, np.newaxis] + x[inds, :]
    return rand * sigmas + x[inds]
###########

def sample(size, K):
    return mixture_rvs(size, weights, mu, sd * np.ones(weights.shape[0]))
