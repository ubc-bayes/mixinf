# suite of functions for doing vanilla hamiltonian monte carlo

# preamble
#import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.linalg import sqrtm
import cvxpy as cp
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io



def HMC(logp, sp, K, epsilon, L, T, burnin = 0.25, p0 = None, verbose = False):
    # adapted from Radford Neal; see https://arxiv.org/pdf/1206.1901.pdf p.14
    # logp is the target log density, sp is its score function
    # K is the dimension of the support of logp (i.e. dimension of problem)
    # epsilon is the leapfrog integrator's step size
    # L is the number of steps to run the leapfrog integrator for
    # T is the number of steps to run the chain for
    # burnin is a float between (0,1) with the proportion of chain to burn in
    # p0 is the initial point of the sample; initialized from a N(0,sqrt(10)) if not specified
    # verbose is a boolean indicating whether to print messages along the way
    #
    # returns a size(T - round(burnin*T), K) array with samples from p

    # define potential energy function and gradient
    if verbose: print('getting potential energy and gradient')
    U = lambda x : -logp(x)
    gradU = lambda x : -sp(x)

    # initialize sample from N(0,10I) if not provided
    if p0 is None:
        p = np.zeros(K)
    else:
        p = p0
    if verbose: print('initial sample: ' + str(p))

    # get total number of iterations
    maxiter = int(np.round(T/(1-burnin)))

    # init array with sample
    ps = np.zeros((maxiter+1,K))
    ps[0,:] = p

    if verbose: print('running chain')
    if verbose: print()
    # run chain
    for t in range(maxiter):
        if t < maxiter - T:
            print(str(t+1) + '/' + str(maxiter - T) + ' (burn-in)', end = '\r')
        else:
            print(str(t+1 - maxiter + T) + '/' + str(T) + ' (sampling)', end = '\r')

        # sample momentum variables from standard normal
        m = np.random.randn(K)

        # save initial samples
        p = ps[t,:]
        p_init = p
        m_init = m

        # take half-step for momentum
        if verbose: print('talking first momentum half-step')
        m -= epsilon * gradU(p) / 2

        # do leapfrog integration
        if verbose: print('doing leapfrog iteration')
        for i in range(L):
            # step for position
            p += epsilon*m
            # step for momentum, except at end of trajectory
            if i < L-1:
                m -= epsilon * gradU(p)
        # end for
        if verbose: print('proposal: ' + str(p))

        # take half-step for momentum at the end and negate momentum
        if verbose: print('take last momentum half-step')
        m -= epsilon * gradU(p) / 2
        m = -m

        # calculate initial and final energies
        if verbose: print('calculating energies')
        U_init = U(p_init)
        U_final = U(p)
        K_init = 0.5 * np.sum(m_init**2)
        K_final = 0.5 * np.sum(m**2)

        # accept/reject proposal
        if verbose: print('accept/reject step')
        if verbose: print('hastings ratio: ' + str(np.minimum(1, np.exp(U_init - U_final + K_init - K_final))))
        if np.random.rand(1) < np.minimum(1, np.exp(U_init - U_final + K_init - K_final)):
            if verbose: print('proposal acepted!')
            ps[t+1,:] = p
        else:
            ps[t+1,:] = p_init

        if verbose: print()
    # end for

    # do burn-in
    ps = ps[-T:,:]
    return ps
