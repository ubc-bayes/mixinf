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



def LogSumExp(x):
    maxx = np.maximum(x)
    return maxx + np.log(np.sum(np.exp(x-maxx)))


def mixture_sample(size, mus, sqrtSigmas, alphas):

    # init everything to account only for positive weights
    K = mus.shape[1]
    options = np.atleast_1d(alphas > 0)
    alphas = alphas[options]
    mus = mus[options,:]
    # adjust if diagonal matrix is provided
    if sqrtSigmas.ndim == 2: sqrtSigmas = np.matmul(np.eye(K), sqrtSigmas[:,:,np.newaxis])
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
        mu = mus[k,:]

        # check if Sigmas is 3dim (full) or 2dim (diagonal)
        # and build matrix correspondingly
        if Sigmas.ndim == 2:
            Sigma = np.squeeze(Sigmas[k,:])
            SigmaSqrt = np.sqrt(Sigma)
            logdet = np.log(np.prod(Sigma))

            # density
            out += alphas[k]*np.exp(-0.5*K*np.log(2*np.pi) - 0.5*logdet - 0.5*np.sum(((x-mu)/SigmaSqrt)**2, axis=-1))
        else:
            Sigma = Sigmas[k,:,:]
            invSigma = np.linalg.inv(Sigma)
            sign, logdet = np.linalg.slogdet(Sigma)

            # density
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
        plt.savefig(plotpath + 'alpha_traceplot_' + str(iteration) + '.jpg', dpi = 300)
        plt.clf()

    if verbose: print('alpha optimized in ' + str(k+1) + ' iterations')

    return alpha



def new_gaussian(logp, K, diagonal = False, mu0 = None, var0 = None, gamma_init = None, B = 1000, maxiter = 500, tol = 0.001, verbose = True, traceplot = True, plotpath = 'plots/', iteration = 1):

    if gamma_init is None:
        gamma_init = lambda k : 0.01/np.sqrt(k+1)

    # init values at inflated std normal if not specified
    if mu0 is None:
        mu = np.zeros((1,K))
    else:
        mu = mu0

    if var0 is None: var0 = 3
    if diagonal:
        Sigma = var0 * np.ones(K)
        SigmaLogDet = np.log(np.prod(Sigma))
        SigmaInv = np.ones(K) / var0
        SigmaSqrt = np.sqrt(var0) * np.ones(K)
    else:
        Sigma = var0*np.eye(K)
        SigmaLogDet = K*np.log(var0)
        SigmaInv = np.eye(K)/var0
        SigmaSqrt = np.sqrt(var0)*np.eye(K)


    init_obj = np.array([])
    convergence = False

    for k in range(maxiter):
        if verbose: print('iteration: ' + str(k+1))

        # assess convergence
        if verbose: print('assessing convergence')
        if convergence: break

        # current approximation
        if verbose: print('building current approx')
        if diagonal:
            logq = lambda x : -0.5*K*np.log(2*np.pi) - 0.5*SigmaLogDet - 0.5*np.sum(((x-mu)/SigmaSqrt)**2, axis=-1)
            sample_q = lambda size : mu + np.random.randn(size,K) * SigmaSqrt
        else:
            logq = lambda x : -0.5*K*np.log(2*np.pi) - 0.5*SigmaLogDet - 0.5*((x-mu).dot(SigmaInv)*(x-mu)).sum(axis=-1)
            sample_q = lambda size : mu + np.random.randn(size,K)@SigmaSqrt

        # estimate gradients
        if verbose: print('estimating gradients')
        qs = sample_q(B)
        logp1 = (1 + logq(qs) - logp(qs)) # relevant quantity

        if diagonal:
            grad_mu = np.mean(logp1[:,np.newaxis]*((qs-mu)/Sigma), axis=0)
            grad_Sigma = np.mean(0.5*logp1[:,np.newaxis]*SigmaInv*(((qs-mu)/SigmaSqrt)**2 - 1), axis=0)
        else:
            matprod = np.matmul(SigmaInv + SigmaInv.T, (qs-mu)[:,:,np.newaxis])
            grad_mu = (0.5*logp1.reshape(B,1)*np.squeeze(matprod, axis=-1)).mean(axis=0)
            grad_Sigma = 0.5 * np.matmul(SigmaInv.T, (logp1[:,np.newaxis,np.newaxis] * np.matmul(np.matmul((qs-mu)[:,:,np.newaxis], (qs-mu)[:,np.newaxis,:]), SigmaInv))).mean(axis=0)

        if verbose: print('mu gradient: ' + str(grad_mu))
        if verbose: print('Sigma gradient: ' + str(grad_Sigma))

        # update estimates
        if verbose: print('updating estimates')
        mu -= gamma_init(k)*grad_mu
        if verbose: print('new mu: ' + str(mu))
        Sigma -= gamma_init(k)*grad_Sigma
        if not diagonal and not np.all(np.linalg.eigvals(Sigma) > 0):
            print('covariance matrix not positive definite, taking absolute value')
            Sigma = np.abs(np.linalg.det(Sigma))*np.eye(K)
        if diagonal: Sigma = np.maximum(0, Sigma) # if diagonal, don't let negative entries exist
        Sigma = np.minimum(Sigma, 1e2) # don't let variance explote
        if verbose: print('new Sigma: ' + str(Sigma))

        # update covariance matrix according to dimension and whether it's diagonal
        if K == 1:
            # this applies to both diagonal and non diagonal matrices
            SigmaLogDet = np.log(np.squeeze(Sigma))
            SigmaInv = 1/Sigma
            SigmaSqrt = np.sqrt(Sigma)
        elif K==2 and not diagonal:
            # calculate determinant etc by hand
            SigmaLogDet = np.log(Sigma[0,0]*Sigma[1,1] - Sigma[0,1]*Sigma[1,0])
            SigmaInv = np.array([[Sigma[1,1], -Sigma[0,1]], [-Sigma[1,0], Sigma[1,1]]])/np.exp(SigmaLogDet)
            SigmaSqrt = sqrtm(Sigma)
        elif K>2 and not diagonal:
            sign, SigmaLogDet = np.linalg.slogdet(Sigma)
            SigmaInv = np.linalg.inv(Sigma)
            SigmaSqrt = sqrtm(Sigma)

        if K >1 and diagonal:
            # O(K) calculating this
            SigmaLogDet = np.log(np.prod(Sigma))
            SigmaInv = 1/Sigma
            SigmaSqrt = np.sqrt(Sigma)


        # estimate objective
        if traceplot:
            init_obj = np.append(init_obj, KL(logq, sample_q, logp, 100000))
            if verbose: print('new KL: ' + str(init_obj[-1]))

        # update convergence
        if verbose: print('updating convergence')
        if np.maximum(np.linalg.norm(grad_mu), np.linalg.norm(grad_Sigma)) < tol: convergence = True

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
        plt.savefig(plotpath + 'new_component_traceplot_' + str(iteration) + '.jpg', dpi = 300)
        plt.clf()

    # if diagonal, return vectors, else return matrices
    if diagonal:
        return mu.reshape(K), Sigma.reshape(K), SigmaSqrt.reshape(K), SigmaLogDet, SigmaInv.reshape(K)
    else:
        return mu.reshape(K), Sigma.reshape(K,K), SigmaSqrt.reshape(K,K), SigmaLogDet, SigmaInv.reshape(K,K)
# end



def bvi(logp, N, K, regularization = None, gamma_init = None, gamma_alpha = None, B = 1000, verbose = True, traceplot = True, plotpath = 'plots/'):

    if regularization is None:
        regularization = lambda k : 1/np.sqrt(k+2)

    # initialize
    if verbose: print('getting initial approximation')
    mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logp, K, diagonal = False, gamma_init = gamma_init, B = B, maxiter = 1000, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath)
    if verbose: print('initial mean: ' + str(mu))
    if verbose: print('initial variance: ' + str(Sigma))

    def logq(x): return -0.5*K*np.log(2*np.pi) - 0.5*SigmaLogDet - 0.5*((x-mu).dot(SigmaInv)*(x-mu)).sum(axis=-1)
    def sample_q(size): return mu + np.random.randn(size,K)@SigmaSqrt

    # init values
    mus = np.zeros((N,K))
    mus[0:,] = mu

    Sigmas = np.zeros((N,K,K))
    Sigmas[0,:,:] = Sigma

    sqrtSigmas = np.zeros((N,K,K))
    sqrtSigmas[0,:,:] = SigmaSqrt

    alphas = np.zeros(N)
    alphas[0] = 1

    objs = np.zeros(N)
    objs[0] = KL(logq, sample_q, logp, 100000)
    if verbose: print('KL to target: ' + str(objs[0]))

    # bvi loop
    print()
    for iter_no in range(1, N):
        if verbose: print('iteration ' + str(iter_no+1))

        # draw initial guess from inflation of current mixture
        inflation = np.random.poisson(size=(1,K)) + 1
        mu_guess = inflation*sample_q(1)
        Sigma_guess = inflation[0,0]*np.amax(np.diagonal(Sigma))
        Sigma_guess = 3

        # get new gaussian
        if verbose: print('obtaining new component')
        logresidual = lambda x : (logp(x) - logq(x)) / regularization(iter_no)
        try:
            mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logresidual, K,  diagonal = False, mu0 = mu_guess, var0 = Sigma_guess, gamma_init = gamma_init, B = B, maxiter = 1000, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath, iteration = iter_no+1)
        except:
            if verbose: print('new component optimization failed, setting to previous component')

        # update mean
        mus[iter_no,:] = mu
        if verbose: print('new mean: ' + str(mu))

        # update sigma and get sqrt
        Sigmas[iter_no,:,:] = Sigma
        sqrtSigmas[iter_no,:,:] = SigmaSqrt
        if verbose: print('new covariance: ' + str(Sigma))

        # define new component logpdf and sampler
        def logh(x): return -0.5*K*np.log(2*np.pi) - 0.5*SigmaLogDet - 0.5*(np.dot((x-mu), SigmaInv)*(x-mu)).sum(axis=-1)
        def sample_h(size): return mu + np.matmul(np.random.randn(size, K), SigmaSqrt)

        # optimize weights
        if verbose: print('optimizing weights')
        try:
            alpha = update_alpha(logq, sample_q, logh, sample_h, logp, K, gamma_alpha = gamma_alpha, B = B, verbose = verbose, traceplot = traceplot, plotpath = plotpath, maxiter = 1000, tol = 1e-10, iteration = iter_no+1)
            if np.isnan(alpha):
                print('weight is NaN; setting new weight to 0')
                alpha = 0
        except:
            print('weight optimization failed; setting new weight to 0')
            alpha = 0
        alphas[0:iter_no] = (1-alpha)*alphas[0:iter_no]
        alphas[iter_no] = alpha
        if verbose: print('new weight: ' + str(alpha))

        # define new mixture
        logq = lambda x : mixture_logpdf(x, mus[0:iter_no,:], Sigmas[0:iter_no,:,:], alphas[0:iter_no])
        sample_q = lambda size : mixture_sample(size, mus, sqrtSigmas, alphas)

        # estimate kl
        if verbose: print('estimating new KL')
        objs[iter_no] = KL(logq, sample_q, logp, 100000)
        if verbose: print('KL to target: ' + str(objs[iter_no]))


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


def bvi_diagonal(logp, N, K, regularization = None, gamma_init = None, gamma_alpha = None, B = 1000, verbose = True, traceplot = True, plotpath = 'plots/'):

    if regularization is None:
        regularization = lambda k : 1/np.sqrt(k+2)

    # initialize
    if verbose: print('getting initial approximation')
    mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logp, K, diagonal = True, gamma_init = gamma_init, B = B, maxiter = 1000, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath)
    if verbose: print('initial mean: ' + str(mu))
    if verbose: print('initial variance: ' + str(Sigma))

    def logq(x): return -0.5*K*np.log(2*np.pi) - 0.5*SigmaLogDet - 0.5*np.sum(((x-mu)/SigmaSqrt)**2, axis=-1)
    def sample_q(size): return mu + np.random.randn(size,K)*SigmaSqrt

    # init values
    mus = np.zeros((N,K))
    mus[0:,] = mu

    Sigmas = np.zeros((N,K))
    Sigmas[0,:,] = Sigma

    sqrtSigmas = np.zeros((N,K))
    sqrtSigmas[0,:] = SigmaSqrt

    alphas = np.zeros(N)
    alphas[0] = 1

    objs = np.zeros(N)
    objs[0] = KL(logq, sample_q, logp, 100000)
    if verbose: print('KL to target: ' + str(objs[0]))

    # bvi loop
    print()
    for iter_no in range(1, N):
        if verbose: print('iteration ' + str(iter_no+1))

        # draw initial guess from inflation of current mixture
        inflation = np.random.poisson(size=(1,K)) + 1
        mu_guess = inflation*sample_q(1)
        Sigma_guess = inflation[0,0]*np.amax(Sigma)
        Sigma_guess = 3

        # get new gaussian
        if verbose: print('obtaining new component')
        logresidual = lambda x : (logp(x) - logq(x)) / regularization(iter_no)
        try:
            mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logresidual, K,  diagonal = True, mu0 = mu_guess, var0 = Sigma_guess, gamma_init = gamma_init, B = B, maxiter = 1000, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath, iteration = iter_no+1)
        except:
            if verbose: print('new component optimization failed, setting to previous component')

        # update mean
        mus[iter_no,:] = mu
        if verbose: print('new mean: ' + str(mu))

        # update sigma and get sqrt
        Sigmas[iter_no,:] = Sigma
        sqrtSigmas[iter_no,:] = SigmaSqrt
        if verbose: print('new covariance: ' + str(Sigma))

        # define new component logpdf and sampler
        def logh(x): return -0.5*K*np.log(2*np.pi) - 0.5*SigmaLogDet - 0.5*np.sum(((x-mu)/SigmaSqrt)**2, axis=-1)
        def sample_h(size): return mu + np.random.randn(size, K)*SigmaSqrt

        # optimize weights
        if verbose: print('optimizing weights')
        try:
            alpha = update_alpha(logq, sample_q, logh, sample_h, logp, K, gamma_alpha = gamma_alpha, B = B, verbose = verbose, traceplot = traceplot, plotpath = plotpath, maxiter = 1000, tol = 1e-10, iteration = iter_no+1)
            if np.isnan(alpha):
                print('weight is NaN; setting new weight to 0')
                alpha = 0
        except:
            print('weight optimization failed; setting new weight to 0')
            alpha = 0
        alphas[0:iter_no] = (1-alpha)*alphas[0:iter_no]
        alphas[iter_no] = alpha
        if verbose: print('new weight: ' + str(alpha))

        # define new mixture
        logq = lambda x : mixture_logpdf(x, mus[0:iter_no,:], Sigmas[0:iter_no,:], alphas[0:iter_no])
        sample_q = lambda size : mixture_sample(size, mus, sqrtSigmas, alphas)

        # estimate kl
        if verbose: print('estimating new KL')
        objs[iter_no] = KL(logq, sample_q, logp, 100000)
        if verbose: print('KL to target: ' + str(objs[iter_no]))


        if verbose: print()
    # end for

    if verbose: print('done!')
    active = alphas > 0
    if verbose: print('means: ' + str(np.squeeze(mus[active])))
    if verbose: print('sigmas: ' + str(np.squeeze(Sigmas[active])))
    if verbose: print('weights: ' + str(np.squeeze(alphas[active])))
    if verbose: print('KL: ' + str(objs[iter_no]))

    iter_no += 1

    return mus[0:iter_no,:], Sigmas[0:iter_no,:], alphas[0:iter_no], objs[0:iter_no]
