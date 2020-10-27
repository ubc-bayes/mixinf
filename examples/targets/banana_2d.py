# returns banana 2-dim density and sampler
import numpy as np
from scipy.special import gamma
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt

# TODO: MAKE A SAMPLER
# see http://probability.ca/jeff/ftpdir/adaptex.pdf

# AUXILIARY FUNCTIONS ####
###########
def norm_logpdf(x, loc = np.array([0]).reshape(1, 1), scale = np.array([1])):
    """
    evaluate isotropic normal logpdf at x with mean loc and sd scale

    - x is an (m+1)d array, where the last dimension accounts for multivariate x
        eg x[0, 0,..., 0, :] is the first observation and has shape K
    - loc is a shape(N, K) array
    - scale is a shape(N,). The covariance matrix is given by scale[i]**2 * np.eye(N) (ie Gaussians are isotropic)

    returns an md array with same shapes as x (except the last dimension)
    """
    K = x.shape[-1]
    loc = loc.T
    return -0.5 * ((x[..., np.newaxis] - loc)**2).sum(axis = -2) / scale**2 - 0.5 * K *  np.log(2 * np.pi) - K * np.log(scale)
###########


###########
def phi(x, b):
    """
    this is used to define the banana distribution
    for x in Rd, phi(x) = [x0 / 100, x1 + b*x0^2 - 100*b, x2, ...]
    """
    if x.shape[0] == 1:
        return x / 100

    y = x
    y[..., 0] = x[..., 0] / 100
    y[..., 1] = x[..., 1] + b*x[..., 0]**2 - 100*b
    return y
###########


# CREATE DENSITY ####
b = 0.1
def p_aux(x, K): return norm_logpdf(phi(x, b), loc = np.zeros((1, K)), scale = np.ones(1))


# CREATE SAMPLER ####
def sample(size, K):
    if K != 2:
        print('error: banana dist is only 2-dim')
        return np.zeros((size, K))



    return size


def p(x): return p_aux(x, 2)

len = 100
xx = np.linspace(-20, 20, len)
yy = np.linspace(-30, 10, len)
tt = np.array(np.meshgrid(xx, yy)).T.reshape(len**2, 2)
#print(tt.shape, tt)
zz = p(tt).reshape(len, len)

for i in np.arange(len):
    for j in np.arange(len):
        print(np.array([xx[i], yy[j]]))
        zz[i, j] = p(np.array(xx[i], yy[j]).reshape(1, 2))


#print(zz.shape)
plt.contour(xx, yy, zz)
plt.savefig('banana.png')
