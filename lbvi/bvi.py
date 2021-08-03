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
    if sqrtSigmas.ndim == 2: sqrtSigmas = sqrtSigmas[:,np.newaxis]*np.eye(K)
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


def KL(logq, sample_q, logp, B = 100000):
    theta = sample_q(B)
    return np.abs(np.mean(logq(theta) - logp(theta)))



def ksd(logp, sample_q, up, B = 1000):
    """
    estimate ksd

    inputs:
        - p target logdensity
        - sample_q is a function that generates samples from the variational approximation
        - up function to calculate expected value of
        - B number of MC samples
    outputs:
        - scalar, the estimated ksd
    """

    # generate samples
    X = sample_q(2*B) # sample from mixture

    Y = X[-B:, :]
    X = X[:B, :]

    return np.abs(up(X, Y).mean())


def update_alpha(logq, sample_q, logh, sample_h, logp, K, gamma_alpha = None, B = 1000, verbose = False, traceplot = True, plotpath = 'plots/', maxiter = 500, tol = 0.01, iteration = 1):

    if gamma_alpha is None:
        gamma_alpha = lambda k : 0.01/np.sqrt(k+1)

    alpha = 0
    convergence = False
    alpha_objs = np.array([])

    for k in range(maxiter):
        if verbose: print(str(k) + '/' + str(maxiter), end = '\r')

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

        #if verbose: print(':', end='')

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
        if verbose:
            print('iteration: ' + str(k+1))
        else:
            print(str(k) + '/' + str(maxiter), end = '\r')

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
        logarithmp1 = 1 + logq(qs) - logp(qs) # relevant quantity

        if np.isnan(logarithmp1).any():
            print('nans in logarithmp1 term; breaking')
            print('iteration: ' + str(k+1))
            print('mean: ' + str(mu))
            print('covariance: ' + str(Sigma))
            print('Sqrt covariance: ' + str(SigmaSqrt))
            print('SigmaInv: ' + str(SigmaInv))
            print('mc sample: ' + str(np.squeeze(qs)))
            print('logq: ' + str(logq(qs)))
            print('logp: ' + str(logp(qs)))
            break

        if diagonal:
            grad_mu = np.mean(logarithmp1[:,np.newaxis]*((qs-mu)/Sigma), axis=0)
            grad_Sigma = np.mean(0.5*logarithmp1[:,np.newaxis]*SigmaInv*(((qs-mu)/SigmaSqrt)**2 - 1), axis=0)
        else:
            matprod = np.matmul(SigmaInv + SigmaInv.T, (qs-mu)[:,:,np.newaxis])
            grad_mu = (0.5*logarithmp1.reshape(B,1)*np.squeeze(matprod, axis=-1)).mean(axis=0)
            grad_Sigma = 0.5 * np.matmul(SigmaInv.T, (logarithmp1[:,np.newaxis,np.newaxis] * np.matmul(np.matmul((qs-mu)[:,:,np.newaxis], (qs-mu)[:,np.newaxis,:]), SigmaInv))).mean(axis=0)

        if verbose: print('mu gradient: ' + str(grad_mu))
        if verbose: print('Sigma gradient: ' + str(grad_Sigma))

        # update estimates
        if verbose: print('updating estimates')
        if np.isinf(grad_mu):
            print('infinite mu gradient; breaking')
            break
        mu -= gamma_init(k)*grad_mu
        if verbose: print('new mu: ' + str(mu))

        if np.isinf(grad_Sigma):
            print('infinite Sigma gradient; breaking')
            break
        Sigma -= gamma_init(k)*grad_Sigma

        if not diagonal and not np.all(np.linalg.eigvals(Sigma) > 0):
            print('covariance matrix not positive definite, taking absolute value')
            Sigma = np.abs(np.linalg.det(Sigma))*np.eye(K)
        if diagonal: Sigma = np.maximum(1e-5, Sigma) # if diagonal, don't let negative or 0 entries exist
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



def choose_kernel(y, logp, logq = None, sample_q = None, verbose = False):
    """
    y is (N,K) with sample locations, logp target logdensity, logq,sample_q current approximation logdensity and sampler
    returns (1,K) array
    """
    N = y.shape[0]
    K = y.shape[1]

    kls = np.zeros(N)
    for n in range(N):
        if verbose: print(str(n+1) + '/' + str(N), end = '\r')

        # new component is a N(y_n, 9I)
        mu = y[n,:].reshape(K)
        tmp_logh = lambda x : -0.5*K*np.log(2*np.pi) - 0.5*K*np.log(9) - 0.5*np.sum(((x-mu)/3)**2, axis=-1)
        tmp_sample_h = lambda size : y[n,:] + 3*np.random.randn(size,K)

        if logq is None:
            # for first iteration, calculate individual KL divergences
            # between target and components with N(y_n, 9I)
            kls[n] = KL(logq = tmp_logh, sample_q = tmp_sample_h, logp = logp, B = 10000)
        else:
            # for other iterations, add each component to mixture with small weight
            # and calculate KL
            def tmp_logq(x):
                m = np.maximum(np.log(0.95) + logq(x), np.log(0.05) + tmp_logh(x))
                return m + np.log(np.exp(np.log(0.95) + logq(x) - m) + np.exp(np.log(0.05) + tmp_logh(x) - m))
            def tmp_sample_q(size):
                out = np.zeros((size,K))
                sizeq = int(size*0.95)
                sizeh = size - sizeq
                out[:sizeq] = sample_q(sizeq)
                out[sizeq:] = tmp_sample_h(sizeh)
                return out

            kls[n] = KL(logq = tmp_logq, sample_q = tmp_sample_q, logp = logp, B = 10000)
    # end for
    return y[np.argmin(kls),:].reshape(1,K)



def bvi(logp, N, K, regularization = None, gamma_init = None, gamma_alpha = None, maxiter_alpha = 1000, maxiter_init = 1000, B = 1000, tol = 0.0001, verbose = True, traceplot = True, plotpath = 'plots/', stop_up = None):

    if verbose:
        print('running boosting black-box variational inference')
        print()

    t0 = time.perf_counter()

    if regularization is None:
        regularization = lambda k : 1/np.sqrt(k+2)

    # initialize
    convergence = False
    if verbose: print('choosing starting point')
    y_init = choose_kernel(y, logp, logq = None, sample_q = None, verbose = True)
    print('chosen point: ' + str(y_init))

    if verbose: print('getting initial approximation')
    mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logp, K, diagonal = False, gamma_init = gamma_init, B = B, maxiter = maxiter_init, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath)
    if verbose:
        print('initial mean: ' + str(mu))
        print('initial variance: ' + str(Sigma))

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

    obj_timer0 = time.perf_counter()
    # estimate kl and ksd
    kls = np.array([KL(logq, sample_q, logp, 100000)])
    obj = np.inf
    objs = None
    if verbose: print('KL to target: ' + str(kls[0]))

    if stop_up is not None:
        objs = np.array([ksd(logp, sample_q, stop_up, B = 100000)])
        if verbose: print('KSD to target: ' + str(objs[0]))
    obj_timer = time.perf_counter() - obj_timer0

    # init cpu time and active kernels
    active_kernels = np.array([1.])
    cpu_time = np.array([time.perf_counter() - obj_timer - t0])

    if verbose:
        print('cumulative cpu time: ' + str(cpu_time[-1]))
        print()

    ############
    ############
    # bvi loop #
    ############
    ############
    for iter_no in range(1, N):
        if verbose: print('iteration ' + str(iter_no+1))
        if convergence: break

        # draw initial guess from provided sample
        if verbose: print('choosing new initialization point')
        mu_guess = choose_kernel(y, logp, logq, sample_q, verbose = True)
        Sigma_guess = 3
        if verbose: print('initializing at ' + str(mu_guess))

        # get new gaussian
        if verbose: print('obtaining new component')
        logresidual = lambda x : (logp(x) - logq(x)) / regularization(iter_no)
        try:
            mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logresidual, K,  diagonal = False, mu0 = mu_guess, var0 = Sigma_guess, gamma_init = gamma_init, B = B, maxiter = maxiter_init, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath, iteration = iter_no+1)
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
            alpha = update_alpha(logq, sample_q, logh, sample_h, logp, K, gamma_alpha = gamma_alpha, B = B, verbose = verbose, traceplot = traceplot, plotpath = plotpath, maxiter = maxiter_alpha, tol = 1e-10, iteration = iter_no+1)
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

        # estimate divergence
        obj_timer0 = time.perf_counter()
        # estimate kl and ksd
        kls = np.append(kls, KL(logq, sample_q, logp, 100000))
        if verbose: print('KL to target: ' + str(kls[-1]))

        if stop_up is not None:
            objs = np.append(objs, ksd(logp, sample_q, stop_up, B = 100000))
            if verbose: print('KSD to target: ' + str(objs[-1]))
        obj_timer = time.perf_counter() - obj_timer0

        # assess convergence
        if stop_up is not None:
            if objs[-1] < tol: convergence = True
        else:
            if kls[-1] < tol: convergence = True

        # calculate cumulative computing time and active kernels
        if alpha == 0:
            active_kernels = np.append(active_kernels, active_kernels[-1])
        else:
            active_kernels = np.append(active_kernels, active_kernels[-1] + 1)
        cpu_time = np.append(cpu_time, time.perf_counter() - obj_timer - t0)

        if verbose:
            print('number of active kernels: ' + str(active_kernels[-1]))
            print('cumulative cpu time: ' + str(cpu_time[-1]))
            print()
    # end for

    active = alphas > 0
    if not convergence: iter_no += 1 # if it converged, assessment was done in next iterations and we need to decrease
    if verbose:
        print('done!')
        print('number of active kernels: ' + str(active_kernels[-1]))
        print('means: ' + str(np.squeeze(mus[active])))
        print('sigmas: ' + str(np.squeeze(Sigmas[active])))
        print('weights: ' + str(np.squeeze(alphas[active])))
        print('KL: ' + str(kls[-1]))
        if stop_up is not None: print('KSD: ' + str(objs[-1]))


    return mus[0:iter_no,:], Sigmas[0:iter_no,:,:], alphas[0:iter_no], objs, cpu_time, active_kernels, kls


def bvi_diagonal(logp, N, K, regularization = None, gamma_init = None, gamma_alpha = None, maxiter_alpha = 1000, maxiter_init = 1000, B = 1000, tol = 0.0001, verbose = True, traceplot = True, plotpath = 'plots/', stop_up = None, y = None):

    if verbose:
        print('running boosting black-box variational inference with diagonal covariance matrix')
        print()
    yyyy = np.copy(y)

    t0 = time.perf_counter()

    if regularization is None:
        regularization = lambda k : 1/np.sqrt(k+2)

    # initialize
    convergence = False
    if verbose: print('choosing starting point')
    y_init = choose_kernel(y, logp, logq = None, sample_q = None, verbose = True)
    print('chosen point: ' + str(y_init))

    if verbose: print('getting initial approximation')
    mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logp, K, diagonal = True, mu0 = y_init, gamma_init = gamma_init, B = B, maxiter = maxiter_init, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath)
    if verbose:
        print('initial mean: ' + str(mu))
        print('initial variance: ' + str(Sigma))

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

    obj_timer0 = time.perf_counter()
    # estimate kl and ksd
    kls = np.array([KL(logq, sample_q, logp, 100000)])
    if verbose: print('KL to target: ' + str(kls[0]))
    obj = np.inf
    objs = None
    if stop_up is not None:
        objs = np.array([ksd(logp, sample_q, stop_up, B = 1000)])
        if verbose: print('KSD to target: ' + str(objs[0]))
    obj_timer = time.perf_counter() - obj_timer0

    # init cpu time and active kernels
    active_kernels = np.array([1.])
    cpu_time = np.array([time.perf_counter() - obj_timer - t0])


    if verbose:
        print('cumulative cpu time: ' + str(cpu_time[-1]))
        print()

    ############
    ############
    # bvi loop #
    ############
    ############
    for iter_no in range(1, N):
        if verbose: print('iteration ' + str(iter_no+1))
        if convergence: break
        y = np.copy(yyyy)

        # draw initial guess from provided sample
        if verbose: print('choosing new initialization point')
        mu_guess = choose_kernel(y, logp, logq, sample_q, verbose = True)
        Sigma_guess = 3
        if verbose: print('initializing at ' + str(mu_guess))

        # get new gaussian
        if verbose:
            print('obtaining new component')
            print('regularization: ' + str(regularization(iter_no)))
        logresidual = lambda x : (logp(x) - logq(x)) / regularization(iter_no)
        #print('logq(1) = ' + str(logq(np.ones((1,K)))))
        #print('logp(1) = ' + str(logp(np.ones((1,K)))))
        #print('logresidual(1) = ' + str(logresidual(np.ones((1,K)))))
        try:
            mu, Sigma, SigmaSqrt, SigmaLogDet, SigmaInv = new_gaussian(logresidual, K,  diagonal = True, mu0 = mu_guess, var0 = Sigma_guess, gamma_init = gamma_init, B = B, maxiter = maxiter_init, tol = 0.001, verbose = False, traceplot = traceplot, plotpath = plotpath, iteration = iter_no+1)
            if np.isnan(mu).any() or np.isnan(Sigma).any():
                print('nans in estimates, setting to previous component')
                mu = mus[iter_no-1,:]
                Sigma = Sigmas[iter_no-1,:].reshape(K)
                SigmaSqrt = sqrtSigmas[iter_no-1,:]
                SigmaLogDet = np.log(np.prod(Sigma))
                SigmaInv = 1/Sigma

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
            alpha = update_alpha(logq, sample_q, logh, sample_h, logp, K, gamma_alpha = gamma_alpha, B = B, verbose = verbose, traceplot = traceplot, plotpath = plotpath, maxiter = maxiter_alpha, tol = 1e-10, iteration = iter_no+1)
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

        # estimate divergence
        obj_timer0 = time.perf_counter()
        # estimate kl and ksd
        kls = np.append(kls, KL(logq, sample_q, logp, 100000))
        if verbose: print('KL to target: ' + str(kls[-1]))

        if stop_up is not None:
            objs = np.append(objs, ksd(logp, sample_q, stop_up, B = 1000))
            if verbose: print('KSD to target: ' + str(objs[-1]))
        obj_timer = time.perf_counter() - obj_timer0

        # assess convergence
        #if stop_up is not None:
        #    if objs[-1] < tol: convergence = True
        #else:
        #    if kls[-1] < tol: convergence = True

        # calculate cumulative computing time and active kernels
        if alpha == 0:
            active_kernels = np.append(active_kernels, active_kernels[-1])
        else:
            active_kernels = np.append(active_kernels, active_kernels[-1] + 1)
        cpu_time = np.append(cpu_time, time.perf_counter() - obj_timer - t0)

        if verbose:
            print('number of active kernels: ' + str(active_kernels[-1]))
            print('cumulative cpu time: ' + str(cpu_time[-1]))
            print()
    # end for

    if not convergence: iter_no += 1 # if it did not converge, assessment was done in that iteration and we need to update
    active = alphas > 0
    if verbose:
        print('done!')
        print('number of active kernels: ' + str(active_kernels[-1]))
        print('means: ' + str(np.squeeze(mus[active])))
        print('sigmas: ' + str(np.squeeze(Sigmas[active])))
        print('weights: ' + str(np.squeeze(alphas[active])))
        print('KL: ' + str(kls[-1]))
        if stop_up is not None: print('ksd: ' + str(objs[-1]))


    return mus[0:iter_no,:], Sigmas[0:iter_no,:], alphas[0:iter_no], objs, cpu_time, active_kernels, kls
