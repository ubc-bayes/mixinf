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



def update_mu(logq, logp, theta_0):

    K = theta_0.shape[0]

    # define optimization target
    target = lambda theta : np.log(np.exp(logq(theta)) + 1e-10) - np.log(np.exp(logp(theta)) + 1e-10)

    # solve optimization
    prob = minimize(target, theta_0, method = 'L-BFGS-B')

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
        #mu = np.squeeze(mus[k,:], axis = 0)
        #Sigma = np.squeeze(Sigmas[k,:,:], axis = 0)
        mu = mus[k,:]
        Sigma = Sigmas[k,:,:]
        invSigma = np.linalg.inv(Sigma)

        # density
        sign, logdet = np.linalg.slogdet(Sigma)
        out += alphas[k]*np.exp(-0.5*K*np.log(2*np.pi) - 0.5*logdet - 0.5*(np.dot((x-mu), invSigma)*(x-mu)).sum(axis=-1))

    return np.log(out)


def KL(logq, sample_q, logp, B = 1000):

    theta = sample_q(B)

    return np.mean(logq(theta) - logp(theta), axis=-1)



def update_alpha(logq, sample_q, logh, sample_h, logp, K, gamma_alpha = None, B = 1000, verbose = False, traceplot = True, plotpath = 'plots/', maxiter = 500, tol = 0.01, iteration = 1):

    if gamma_alpha is None:
        gamma_alpha = lambda k : 0.01/np.sqrt(k+1)

    alpha = 0
    convergence = False
    alpha_objs = np.array([])

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
        step = gamma_alpha(k) * grad
        alpha -= step
        alpha = np.maximum(np.minimum(alpha, 1), 0)
        #print('alpha: ' + str(alpha))

        # calculate objective
        if traceplot:
            # define temporary mixture with current alpha and calculate KL
            tmp_logq = lambda x : np.log(alpha*np.exp(logh(x)) + (1-alpha)*np.exp(logq(x)))
            def tmp_sample_q(size):
                out = np.zeros((size,K))
                inds = np.random.rand(size) < alpha
                out[inds,:] = sample_h(np.sum(inds.astype(int)))
                out[~inds,:] = sample_q(size - np.sum(inds.astype(int)))
                return out

            alpha_objs = np.append(alpha_objs, KL(tmp_logq, tmp_sample_q, logp, B))

        # update convergence
        if np.linalg.norm(step) < tol: convergence = True

        if verbose: print(':', end='')

    # end for

    # plot kl trace
    if traceplot:
        plt.clf()
        plt.plot(range(1, alpha_objs.shape[0]+1), alpha_objs, '-k')
        plt.xlabel('iteration')
        plt.ylabel('objective')
        plt.title('KL traceplot')
        plt.savefig(plotpath + 'traceplot_' + str(iteration) + '.jpg', dpi = 300)
        plt.clf()

    if verbose: print('alpha optimized in ' + str(k+1) + ' iterations')

    return alpha



def init_gaussian(logp, K, gamma_init = None, B = 1000, maxiter = 500, tol = 0.001, verbose = True, traceplot = True, plotpath = 'plots/'):

    if gamma_init is None:
        gamma_init = lambda k : 0.01/np.sqrt(k+1)

    # init values at std normal
    mu = np.zeros((1,K))
    var = 3
    init_obj = np.array([])
    convergence = False

    for k in range(maxiter):
        if verbose: print('iteration: ' + str(k+1))

        # assess convergence
        if verbose: print('assessing convergence')
        if convergence: break

        # current approximation
        if verbose: print('building current approx')
        logq = lambda x : -0.5*K*np.log(2*np.pi*var) - 0.5*np.sum(np.power(x-mu, 2), axis = -1) / var
        sample_q = lambda size : mu + np.random.randn(size,K) / np.sqrt(var)

        # estimate gradients
        if verbose: print('estimating gradients')
        qs = sample_q(B)
        logp1 = (1 + logq(qs) - logp(qs)).reshape(B,K) # relevant quantity
        grad_mu = -logp1.mean(axis=0)
        if verbose: print('mu gradient: ' + str(grad_mu))
        grad_var = (1/(2*var)) * (logp1 * (((qs-mu)**2).sum(axis=-1)/var - K)[:,np.newaxis]).mean(axis=0)
        if verbose: print('var gradient: ' + str(grad_var))

        # update estimates
        if verbose: print('updating estimates')
        mu -= gamma_init(k)*grad_mu
        if verbose: print('new mu: ' + str(mu))
        var -= gamma_init(k)*grad_var
        if verbose: print('new var: ' + str(var))

        # estimate objective
        if traceplot:
            init_obj = np.append(init_obj, KL(logq, sample_q, logp, B))
            if verbose: print('new KL: ' + str(init_obj[-1]))

        # update convergence
        if verbose: print('updating convergence')
        if np.maximum(np.linalg.norm(grad_mu), np.linalg.norm(grad_var)) < tol: convergence = True

        #if verbose: print(':', end='')
        if verbose: print()
        # end for

    # plot trace
    if traceplot:
        plt.clf()
        plt.plot(range(1, init_obj.shape[0]+1), init_obj, '-k')
        plt.xlabel('iteration')
        plt.ylabel('objective')
        plt.title('KL traceplot')
        plt.savefig(plotpath + 'traceplot_1.jpg', dpi = 300)
        plt.clf()

    return mu.reshape(K), np.squeeze(var)



def bvi(logp, N, K, regularization = 1., gamma_init = None, gamma_alpha = None, B = 1000, verbose = True, traceplot = True, plotpath = 'plots/'):

    # initialize
    # todo run vi for one iteration to init
    #mu = np.zeros(K) + 0.2
    #Sigma = np.eye(K)
    if verbose: print('getting initial approximation')
    mu, var = init_gaussian(logp, K, gamma_init = gamma_init, B = 10000, maxiter = 5000, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath)
    Sigma = var * np.eye(K)
    if verbose: print('initial mean: ' + str(mu))
    if verbose: print('initial isotropic variance: ' + str(var))

    def logq(x): return -0.5*K*np.log(2*np.pi*var) - 0.5*np.sum(np.power(x-mu, 2), axis = -1) / var
    def sample_q(size): return mu + np.random.randn(size,K) / np.sqrt(var)

    # save values
    mus = np.zeros((N,K))
    mus[0:,] = mu
    Sigmas = np.zeros((N,K,K))
    Sigmas[0,:,:] = Sigma
    sqrtSigmas = np.zeros((N,K,K))
    sqrtSigmas[0,:,:] = sqrtm(Sigma)
    alphas = np.zeros(N)
    alphas[0] = 1
    objs = np.zeros(N)
    objs[0] = KL(logq, sample_q, logp, 100000)
    if verbose: print('KL to target: ' + str(objs[0]))

    # bvi loop
    print()
    for iter_no in range(1, N):
        if verbose: print('iteration ' + str(iter_no+1))

        # draw from inflation of current mixture
        inflation = np.random.poisson(size=(1,K))
        theta_0 = inflation*sample_q(1)

        # optimize mean via BFGS
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
        H = hessian(lambda theta : np.log(np.exp(logq(theta)) + 1e-10) - np.log(np.exp(logp(theta)) + 1e-10))(mu)

        # check that hessian is 2d (sometimes it's nd; this is a feature of autograd's broadcasting)
        if H.ndim == 3: H = H.diagonal(axis1=1, axis2=2)
        if verbose: print('hessian: ' + str(H))

        # try inverting hessian
        try:
            Sigma = 0.5 * regularization * np.linalg.inv(H)
        except:
            print('singular covariance matrix, not updating')
            Sigma = Sigmas[iter_no-1,:,:]
        if not np.all(np.linalg.eigvals(Sigma) > 0):
            print('covariance matrix not positive definite, taking absolute value')
            Sigma = np.abs(Sigma)

        # update sigma and get sqrt
        Sigmas[iter_no,:,:] = Sigma
        sqrtSigma = sqrtm(Sigma)
        sqrtSigmas[iter_no,:,:] = sqrtSigma
        if verbose: print('new covariance: ' + str(Sigma))

        # define new component
        if verbose: print('defining new component')
        def logh(x):
            sign, logdet = np.linalg.slogdet(Sigma)
            return -0.5*K*np.log(2*np.pi) - 0.5*logdet - 0.5*(np.dot((x-mu), 2*H/regularization)*(x-mu)).sum(axis=-1)
        def sample_h(size): return mu + np.matmul(np.random.randn(size, K), sqrtSigma)

        # optimize weights
        if verbose: print('optimizing weights')
        try:
            alpha = update_alpha(logq, sample_q, logh, sample_h, logp, K, gamma_alpha = gamma_alpha, B = B, verbose = verbose, traceplot = traceplot, plotpath = plotpath, maxiter = 250, tol = 1e-10, iteration = iter_no+1)
            if np.isnan(alpha):
                print('weights are NaN; setting new weight to 0')
                alpha = 0
        except:
            print('weight optimization failed; setting new weight to 0')
            alpha = 0
        alphas[0:iter_no] = (1-alpha)*alphas[0:iter_no]
        alphas[iter_no] = alpha
        if verbose: print('new weight: ' + str(alpha))

        # define new mixture
        if verbose: print('building new mixture')
        logq = update_mixture(logh, alpha, logq)
        logq = lambda x : mixture_logpdf(x, mus[0:iter_no,:], Sigmas[0:iter_no,:,:], alphas[0:iter_no])
        sample_q = lambda size : mixture_sample(size, mus, sqrtSigmas, alphas)

        # estimate kl
        if verbose: print('estimating new KL')
        objs[iter_no] = KL(logq, sample_q, logp, 100000)
        if verbose: print('KL to target: ' + str(objs[iter_no]))

        # break if stuck in an error loop where nothing gets updated
        #if np.array_equal(mus[iter_no-1,:], mus[iter_no,:]) and np.array_equal(Sigmas[iter_no-1,:,:], Sigmas[iter_no,:,:]) and alphas[iter_no-1]==alphas[iter_no]:
        #    if verbose: print('no updates from last iteration; breaking')
        #    break

        if verbose: print()
    # end for

    if verbose: print('done!')
    active = alphas > 0
    if verbose: print('means: ' + str(np.squeeze(mus[active])))
    if verbose: print('sigmas: ' + str(np.squeeze(Sigmas[active])))
    if verbose: print('weights: ' + str(np.squeeze(alphas[active])))
    if verbose: print('KL: ' + str(objs[iter_no]))

    iter_no += 1
    return mus[0:iter_no,:], Sigmas[0:iter_no,:,:], alphas[0:iter_no], objs[0:iter_no]
