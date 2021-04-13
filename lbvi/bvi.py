# suite of functions for doing boosting variational inference

# preamble
#import numpy as np
import autograd.numpy as np
from autograd import hessian
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.linalg import sqrtm
import cvxpy as cp
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io



def update_mixture(logh, alpha, old_logq):

    def new_logq(x): return np.log(alpha*np.exp(logh(x)) + (1-alpha)*np.exp(old_logq(x)))

    return new_logq

def update_mu(logq, logp, theta_0):

    K = theta_0.shape[0]

    # define optimization target
    target = lambda theta : np.log(np.exp(logq(theta)) + 1e-10) - np.log(np.exp(logp(theta)) + 1e-10)

    # solve optimization
    prob = minimize(target, theta_0, method = 'BFGS')

    # make sure optimization was done properly
    if not prob.success:
        print('mean optimization failed')
        return theta_0

    return prob.x

def mixture_sample(size, mus, sqrtSigmas, alphas):

    # init everything to account only for positive weights
    K = mus.shape[1]
    options = np.atleast_1d(alphas > 0)
    alphas = alphas[options]
    mus = mus[options,:]
    sqrtSigmas = sqrtSigmas[options,:,:]
    no_options = alphas.shape[0]

    # draw indices to sample from
    inds = np.random.choice(range(no_options), size = size, p = alphas)

    # generate normal deviates
    sample = np.random.randn(size,1,K)
    scaled_sample = np.squeeze(np.matmul(sample, sqrtSigmas[inds,:,:]), axis=1)

    return scaled_sample + mus[inds,:]


def mixture_logpdf(x, mus, Sigmas, alphas):

    N = x.shape[0]
    K = mus.shape[1]
    options = np.atleast_1d(alphas > 0)
    alphas = alphas[options]
    mus = mus[options,:]
    no_options = alphas.shape[0]

    out = np.zeros(N)

    for k in range(no_options):
        # current meand and covariance matrix
        mu = np.squeeze(mus[k,:], axis = 0)
        #Sigma = np.squeeze(Sigmas[k,:,:], axis = 0)
        Sigma = Sigmas[k,:,:]
        invSigma = np.linalg.inv(Sigma)

        # density
        sign, logdet = np.linalg.slogdet(Sigma)
        out += np.exp(-0.5*np.log(2*np.pi) - 0.5*logdet - 0.5*(np.dot((x-mu), invSigma)*(x-mu)).sum(axis=-1))

    return np.log(out)


def update_alpha(logq, sample_q, logh, sample_h, logp, gamma_w = None, B = 1000., maxiter = 500, tol = 0.01):

    if gamma_w is None:
        gamma_w = lambda k : 0.01/np.sqrt(k+1)

    alpha = 0
    convergence = False

    for k in range(maxiter):

        # assess convergence
        if convergence: break

        # generate samples
        hs = sample_h(B)
        #print('hs: ' + str(hs))
        qs = sample_q(B)
        #print('qs: ' + str(qs))

        # estimate gradient
        g_theta = lambda theta : np.log((1-alpha)*np.exp(logq(theta)) + alpha*np.exp(logh(theta))) - logp(theta)
        grad = (g_theta(hs) - g_theta(qs)).mean(axis=0)
        #print('term 1: ' + str(g_theta(hs)))
        #print('term 2: ' + str(g_theta(qs)))
        #print('grad: ' + str(grad))

        # take step
        step = gamma_w(k+1) * grad
        alpha -= step
        alpha = np.maximum(np.minimum(alpha, 1), 0)
        #print('alpha: ' + str(alpha))

        # update convergence
        if np.linalg.norm(step) < tol: convergence = True

    # end for
    return alpha




def bvi(logp, N, K, regularization = 1., gamma_w = None, B = 1000, verbose = True):

    # initialize
    # todo run vi for one iteration to init
    mu = np.zeros(K) + 0.2
    Sigma = np.eye(K)
    def logq(x): return -0.5*np.log(2*np.pi) - 0.5*np.sum(np.power(x-mu, 2), axis = -1)
    def sample_q(size): return mu + np.matmul(np.random.randn(size, K), Sigma)

    # save values
    mus = np.zeros((N,K))
    mus[0:,] = mu
    Sigmas = np.zeros((N,K,K))
    Sigmas[0,:,:] = Sigma
    sqrtSigmas = np.zeros((N,K,K))
    sqrtSigmas[0,:,:] = sqrtm(Sigma)
    alphas = np.zeros(N)
    alphas[0] = 1

    # bvi loop
    for iter_no in range(1, N):
        if verbose: print('iteration ' + str(iter_no+1))

        # draw from current mixture
        theta_0 = sample_q(1)

        # optimize mean
        if verbose: print('optimizing mean')
        try:
            mu = update_mu(logq, logp, theta_0)
        except:
            print('mean optimization failed')
            mu = mus[iter_no,:]
        mus[iter_no,:] = mu
        if verbose:  print('new mean: ' + str(mu))

        # get covariance matrix
        if verbose: print('obtaining new covariance matrix')
        H = -hessian(lambda theta : np.log(np.exp(logq(theta)) + 1e-10) - np.log(np.exp(logp(theta)) + 1e-10))(mu)
        try:
            Sigma = 0.5 * regularization * np.linalg.inv(H)
        except:
            print('singular covariance matrix, not updating')
            Sigma = Sigmas[iter_no-1,:,:]
        if not np.all(np.linalg.eigvals(Sigma) > 0):
            print('covariance matrix not positive definite, not updating')
            Sigma = Sigmas[iter_no-1,:,:]
        Sigmas[iter_no,:,:] = Sigma
        sqrtSigma = sqrtm(Sigma)
        sqrtSigmas[iter_no,:,:] = sqrtSigma
        if verbose: print('new covariance: ' + str(Sigma))

        # define new component
        if verbose: print('defining new component')
        def logh(x):
            sign, logdet = np.linalg.slogdet(Sigma)
            return -0.5*np.log(2*np.pi) - 0.5*logdet - 0.5*(np.dot((x-mu), 2*H/regularization)*(x-mu)).sum(axis=-1)
        def sample_h(size): return mu + np.matmul(np.random.randn(size, K), sqrtSigma)

        # optimize weights
        if verbose: print('optimizing weights')
        try:
            alpha = update_alpha(logq, sample_q, logh, sample_h, logp, gamma_w = None, B = 10, maxiter = 500, tol = 0.01)
            if np.isnan(alpha):
                print('weight optimization failed')
                alpha = 0
        except:
            print('weight optimization failed')
            alpha = 0
        alphas[0:iter_no] = (1-alpha)*alphas[0:iter_no]
        alphas[iter_no] = alpha
        if verbose: print('new weight: ' + str(alpha))

        # define new mixture
        if verbose: print('building new mixture')
        #logq = lambda x : mixture_logpdf(x, mus, Sigmas, alphas)
        logq = update_mixture(logh, alpha, logq)
        sample_q = lambda size : mixture_sample(size, mus, sqrtSigmas, alphas)

        if np.array_equal(mus[iter_no-1,:], mus[iter_no,:]) and np.array_equal(Sigmas[iter_no-1,:,:], Sigmas[iter_no,:,:]) and alphas[iter_no-1]==alphas[iter_no]:
            break

        if verbose: print()
    # end for

    if verbose: print('done!')

    return mus, Sigmas, alphas
