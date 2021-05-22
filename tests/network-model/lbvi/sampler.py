import autograd.numpy as np
from autograd.scipy.special import gammaln, digamma, polygamma, betaln, betainc, erf, erfinv
#import numpy as np
#from scipy.special import gammaln, digamma, polygamma, betaln, betainc, erf, erfinv
import scipy.integrate as integrate
from scipy.linalg import sqrtm
import cProfile, pstats, io
from pstats import SortKey
import pickle as pk
import pandas as pd
import os
import time


################
################
## alpha settings and transforms
################
################

def alpha_lims(alpha):
    # given alpha in (0,1)^S, return mh sampler limits in logit (unconstrained) space
    S = alpha.shape[0]
    lower = np.zeros(S)
    upper = np.zeros(S)
    for s in range(S):
        if alpha[s] < 0.02:
            lower[s] = np.NINF
            upper[s] = np.log(0.02) - np.log1p(-0.02)
        elif alpha[s] < 0.1:
            lower[s] = np.log(0.02) - np.log1p(-0.02)
            upper[s] = np.log(0.1) - np.log1p(-0.1)
        elif alpha[s] < 0.2:
            lower[s] = np.log(0.1) - np.log1p(-0.1)
            upper[s] = np.log(0.2) - np.log1p(-0.2)
        elif alpha[s] < 0.4:
            lower[s] = np.log(0.2) - np.log1p(-0.2)
            upper[s] = np.log(0.4) - np.log1p(-0.4)
        elif alpha[s] < 0.6:
            lower[s] = np.log(0.4) - np.log1p(-0.4)
            upper[s] = np.log(0.6) - np.log1p(-0.6)
        elif alpha[s] < 0.8:
            lower[s] = np.log(0.6) - np.log1p(-0.6)
            upper[s] = np.log(0.8) - np.log1p(-0.8)
        else:
            lower[s] = np.log(0.8) - np.log1p(-0.8)
            upper[s] = np.inf
    return lower, upper

def alpha_stepsize(alpha):
    # given alpha in (0,1)^S, return mh sampler step size
    S = alpha.shape[0]
    steps = np.zeros(S)
    for s in range(S):
        if alpha[s] < 0.02:
            steps[s] = 0.5
        #elif alpha[s] < 0.1:
        #    steps[s] = 0.1
        #elif alpha[s] < 0.2:
        #    steps[s] = 0.04
        #elif alpha[s] < 0.4:
        #    steps[s] = 0.04
        #elif alpha[s] < 0.6:
        #    steps[s] = 0.04
        #elif alpha[s] < 0.8:
        #    steps[s] = 0.04
        else:
            steps[s] = 0.5
    return steps


def constrain_alpha(ualph_prime, ualph_lower, ualph_upper):
    out = np.zeros(ualph_prime.shape[0])

    l = ualph_lower < -100 # first transform on left-most bin
    out[l] = ualph_upper[l] - np.exp(ualph_prime[l])

    u = ualph_upper > 100 # second transform on right-most bin
    out[u] = ualph_lower[u] + np.exp(ualph_prime[u])

    m = np.logical_and(ualph_lower > -100, ualph_upper < 100) # third transform on middle bins
    out[m] = ualph_lower[m] + (ualph_upper[m] - ualph_lower[m])*np.exp(ualph_prime[m])/(1+np.exp(ualph_prime[m]))

    #if ualph_lower < -100: # ==-np.NINF
    #    return ualph_upper - np.exp(ualph_prime)
    #elif ualph_upper > 100: # ==np.inf
    #    return ualph_lower + np.exp(ualph_prime)
    #else:
    #    return ualph_lower + (ualph_upper - ualph_lower)*np.exp(ualph_prime)/(1+np.exp(ualph_prime))

    return out

def unconstrain_alpha(ualph, ualph_lower, ualph_upper):
    out = np.zeros(ualph.shape[0])

    l = ualph_lower < -100 # first transform on left-most bin
    out[l] = np.log(ualph_upper[l] - ualph[l])

    u = ualph_upper > 100 # second transform on right-most bin
    out[u] = np.log(ualph[u] - ualph_lower[u])

    m = np.logical_and(ualph_lower > -100, ualph_upper < 100) # third transform on middle bins
    out[m] = np.log(ualph[m] - ualph_lower[m]) - np.log(ualph_upper[m] - ualph[m])

    #if ualph_lower < -100: # ==-np.NINF
    #    return np.log(ualph_upper - ualph)
    #elif ualph_upper > 100: # ==np.inf
    #    return np.log(ualph - ualph_lower)
    #else:
    #    return np.log((ualph - ualph_lower) / (ualph_upper - ualph))

    return out


def log_jac_ualph(ualph_prime, ualph_lower, ualph_upper):

    out = np.zeros(ualph_prime.shape[0])

    #lu = ualph_lower < -100 or ualph_upper > 100 # lower and upper transforms, same jacobian
    lu = np.logical_or(ualph_lower < -100, ualph_upper > 100) # lower and upper transforms, same jacobian
    out[lu] = ualph_prime[lu]

    # other transform
    out[~lu] = np.log(np.abs(ualph_lower[~lu] + (ualph_upper[~lu] - ualph_lower[~lu])*np.exp(ualph_prime[~lu])/(1+np.exp(ualph_prime[~lu]))**2))

    #if ualph_lower < -100 or ualph_upper > 100: # ==-np.NINF or ==np.inf
    #    return ualph_prime
    #else:
    #    return np.log(np.abs(ualph_lower + (ualph_upper - ualph_lower)*np.exp(ualph_prime)/(1+np.exp(ualph_prime))**2))

    return out

################
################
## nu_int
################
################
def nu_int(x, alph, gam, lamb):

    #out = np.zeros(alph.shape[0])
    #def t1(x, alph, gam, lamb):
    #    if np.floor(lamb) != lamb:
    #        raise ValueError("lamb must be an integer for this method to work")
    #    lamb = int(np.floor(lamb))
    #    lls = np.zeros(lamb)
    #    for n in range(lamb-1):
    #         lls[n] = gammaln(lamb) - gammaln(n+1) - gammaln(lamb-n) - np.log(lamb-n-1.) + np.log(1.-x**(lamb-n-1.)) + np.log(gam) + np.log(lamb)
    #    lls[lamb-1] = np.log(gam) + np.log(lamb) + np.log(-np.log(x))
    #    llmax = lls.max()
    #    lls -= llmax
    #    return np.exp(llmax + np.log((np.exp(lls)*(-1)**(lamb - 1 - np.arange(lamb))).sum()))

    #def t2(x, alpha, gam, lamb):
    #    ll1 = -alph*np.log(x) + (lamb+alph-1.)*np.log1p(-x) - np.log(lamb+alph-1.) - betaln(1.-alph, lamb+alph-1.)
    #    ll2 = np.log(betainc(1.-alph, lamb+alph-1., x))
    #    llmax = max(ll1, ll2)
    #    ll1 -= llmax
    #    ll2 -= llmax
    #    ll = llmax + np.log(np.exp(ll1) + np.exp(ll2))
    #    if ll<0:
    #        print("ll1: ", ll1, "ll2: ", ll2, "llmax: ", llmax, "theta_K: ", x )
    #    m1 = 0
    #    llmax = max(m1, ll)
    #    ll -= llmax
    #    m1 -= llmax
    #    ll = llmax + np.log(np.exp(ll) - np.exp(m1))
    #    ll += np.log(gam) + np.log(lamb) - np.log(alph)
    #    return np.exp(ll)

    #naught = alph == 0
    #out[naught] = t1(x[naught], alph[naught], gam[naught], lamb[naught])

    for s in range(alph.shape[0]):
        if alph[s]==0:
            if np.floor(lamb[s]) != lamb[s]:
                raise ValueError("lamb must be an integer for this method to work")
            lamb[s] = int(np.floor(lamb[s]))
            lls = np.zeros(lamb[s])
            for n in range(lamb[s]-1):
                 lls[n] = gammaln(lamb[s]) - gammaln(n+1) - gammaln(lamb[s]-n) - np.log(lamb[s]-n-1.) + np.log(1.-x[s]**(lamb[s]-n-1.)) + np.log(gam[s]) + np.log(lamb[s])
            lls[lamb[s]-1] = np.log(gam[s]) + np.log(lamb[s]) + np.log(-np.log(x[s]))
            llmax = lls.max()
            lls -= llmax
            return np.exp(llmax + np.log((np.exp(lls)*(-1)**(lamb[s] - 1 - np.arange(lamb[s]))).sum()))
        else:
            ll1 = -alph[s]*np.log(x[s]) + (lamb[s]+alph[s]-1.)*np.log1p(-x[s]) - np.log(lamb[s]+alph[s]-1.) - betaln(1.-alph[s], lamb[s]+alph[s]-1.)
            ll2 = np.log(betainc(1.-alph[s], lamb[s]+alph[s]-1., x[s]))
            llmax = max(ll1, ll2)
            ll1 -= llmax
            ll2 -= llmax
            ll = llmax + np.log(np.exp(ll1) + np.exp(ll2))
            if ll<0:
                print("ll1: ", ll1, "ll2: ", ll2, "llmax: ", llmax, "theta_K: ", x[s] )
            m1 = 0
            llmax = max(m1, ll)
            ll -= llmax
            m1 -= llmax
            ll = llmax + np.log(np.exp(ll) - np.exp(m1))
            ll += np.log(gam[s]) + np.log(lamb[s]) - np.log(alph[s])
            return np.exp(ll)

################
################
## transforms
################
################

def constrain_hyper(ualph, ugam, ulamb):
    # get alpha
    lmax = np.maximum(0, -ualph)
    alph = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-ualph - lmax)))
    # get gamma
    gam = np.exp(ugam)
    # get lambda
    lamb = 1+np.exp(ulamb)
    return alph, gam, lamb

def constrain_rates(uTh):
    # allocate memory for Thetas
    Th = np.zeros((uTh.shape[0], uTh.shape[1]))

    # get Theta[K]
    lmax = np.maximum(0, -uTh[:,-1])
    Th[:,-1] = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1] - lmax)))

    # get Th[1...K-1]
    tmp = np.zeros((3, uTh.shape[0], uTh.shape[1]-1))
    tmp[1, :,:] = -uTh[:,:-1]
    tmp[2, :,:] = -uTh[:,-1][:,np.newaxis]
    lmax = tmp.max(axis=0)
    lnum = lmax + np.log( np.exp(tmp - lmax).sum(axis=0) )
    lmax = tmp[:2,:,:].max(axis=0)
    ldenom = lmax + np.log( np.exp(tmp[:2,:,:] - lmax).sum(axis=0) )
    Th[:,:-1] = np.exp( lnum - ldenom + np.log(Th[:,-1][:,np.newaxis]) )
    return Th

def constrain_rate_k(k, uTh):
    # get Theta[K]
    lmax = np.maximum(0, -uTh[:,-1])
    ThK = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1] - lmax)))

    # get Th[1...K-1]
    tmp = np.zeros((3, uTh.shape[0], 1))
    tmp[1, :,0] = -uTh[:,k]
    tmp[2, :,0] = -uTh[:,-1]
    lmax = tmp.max(axis=0)
    lnum = lmax + np.log( np.exp(tmp - lmax).sum(axis=0) )
    lmax = tmp[:2,:,0].max(axis=0)
    ldenom = lmax + np.log( np.exp(tmp[:2,:,0] - lmax).sum(axis=0) )
    return np.exp( np.squeeze(lnum) - ldenom + np.log(ThK) )

def unconstrain_hyper(alph, gam, lamb):
    ualph = np.log(alph) - np.log1p(-alph)
    ugam = np.log(gam)
    ulamb = np.log(lamb-1.)
    return ualph, ugam, ulamb

def unconstrain_rates(Th):
    uTh = np.zeros((Th.shape[0],Th.shape[1]))
    uTh[:,-1] = np.log(Th[:,-1]) - np.log1p(-Th[:,-1])
    uTh[:,:-1] = np.log(Th[:,:-1]) + np.log1p(-(Th[:,-1])[:,np.newaxis]/Th[:,:-1]) - np.log1p(-Th[:,:-1])
    return uTh



##############################
##############################
##############################
##############################

def log_jac(ualph, ugam, ulamb, uTh):
    # alpha vs ualph jac
    lmax = np.maximum(0, -ualph)
    logjac_alph = -ualph  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-ualph-lmax)))

    # gam vs ugam jac
    logjac_gam = ugam

    # lamb vs ulamb jac
    logjac_lamb = ulamb

    # Th[-1] vs uTh[-1] jac
    lmax = np.maximum(0, -uTh[:,-1])
    logjac_thK = -uTh[:,-1]  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1]-lmax)))

    # this just makes it e^{-tK}/()^1 instead of ()^2
    logjac_thk = logjac_thK + (lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1]-lmax)))

    # Th[1...K-1] vs t[1...K-1] jac
    tmp = np.zeros((2, uTh.shape[0], uTh.shape[1]-1))
    tmp[1, :,:] = -uTh[:,:-1]
    lmax = tmp.max(axis=0)
    logjac_thk = logjac_thk[:,np.newaxis] -uTh[:,:-1] - 2*(lmax + np.log(np.exp(tmp-lmax[np.newaxis,:,:]).sum(axis=0)) )

    # the jacobian matrix is lower triangular, so can just add these up and return
    return logjac_alph + logjac_gam + logjac_lamb + logjac_thK + logjac_thk.sum(axis=-1)


def log_jac_alph(ualph, ugam, ulamb, uTh):
    # alpha vs ualph jac
    lmax = np.maximum(0, -ualph)
    logjac_alph = -ualph  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-ualph-lmax)))
    return logjac_alph

def log_jac_lamb(ualph, ugam, ulamb, uTh):
    # lamb vs ulamb jac
    logjac_lamb = ulamb
    return logjac_lamb


def log_jac_thk(k, ualph, ugam, ulamb, uTh):
    uThK = uTh[:,-1]
    lmax = np.maximum(0, -uThK)
    logjac_thK = -uThK  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uThK-lmax)))

    # this just makes it e^{-tK}/()^1 instead of ()^2
    logjac_thk = logjac_thK + (lmax + np.log(np.exp(0-lmax) + np.exp(-uThK-lmax)))

    # Th[1...K-1] vs t[1...K-1] jac
    uThk = uTh[:,k]
    lmax = np.maximum(0, -uThk)
    logjac_thk = logjac_thk -uThk - 2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uThk-lmax)))
    return logjac_thK + logjac_thk

def log_jac_thK(ualph, ugam, ulamb, uTh):
    # Th[-1] vs uTh[-1] jac
    lmax = np.maximum(0, -uTh[:,-1])
    logjac_thK = -uTh[:,-1]  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1]-lmax)))

    # this just makes it e^{-tK}/()^1 instead of ()^2
    logjac_thk = logjac_thK + (lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1]-lmax)))

    # Th[1...K-1] vs t[1...K-1] jac
    tmp = np.zeros((2, uTh.shape[0], uTh.shape[1]-1))
    tmp[1, :,:] = -uTh[:,:-1]
    lmax = tmp.max(axis=0)
    logjac_thk = logjac_thk[:,np.newaxis] -uTh[:,:-1] - 2*(lmax + np.log(np.exp(tmp-lmax[np.newaxis,:,:]).sum(axis=0)) )

    # the jacobian matrix is lower triangular, so can just add these up and return
    return logjac_thK + logjac_thk.sum(axis=-1)

##############################
##############################
##############################
##############################

################
################
## priors
################
################


def log_prior_alph(alph, a, b):
    return (a-1.)*np.log(alph) + (b-1.)*np.log1p(-alph) - (gammaln(a)+gammaln(b)-gammaln(a+b))

def log_prior_gam(gam, a, b):
    return a*np.log(b) - gammaln(a) + (a-1.)*np.log(gam) - b*gam

def log_prior_lamb(lamb, a, b):
    return a*np.log(b) - gammaln(a) + (a-1.)*np.log(lamb) - b*lamb

def log_prior_th(alph, gam, lamb, Th):
    # the log beta process density
    lp = (-1.-alph[:,np.newaxis])*np.log(Th) + (lamb[:,np.newaxis]+alph[:,np.newaxis]-1.)*np.log1p(-Th)
    # normalizing constant
    lp += np.log(gam[:,np.newaxis]) + gammaln(lamb[:,np.newaxis]+1) - gammaln(lamb[:,np.newaxis]+alph[:,np.newaxis]) - gammaln(1-alph[:,np.newaxis])
    # subtract nu_int at the end
    return lp.sum(axis=-1) - nu_int(Th[:,-1], alph, gam, lamb)

def log_prior_thk(k, alph, gam, lamb, Th):
    # the log beta process density
    lp = (-1.-alph)*np.log(Th[:,k]) + (lamb+alph-1.)*np.log1p(-Th[:,k])
    # subtract nu_int at the end
    return lp

################
################
## likelihood
################
################

def approx_log_like(Edges, Th, N):
    ## add up everything
    #lp = (X*np.log(Th) + np.log(Th)[:,np.newaxis]*X + (N-X)*np.log1p(-Th[:,np.newaxis]*Th)).sum()

    ## remove the diagonal
    #K = X.shape[0]
    #lp -= (X[np.arange(K), np.arange(K)]*2*np.log(Th) + (N-X[np.arange(K), np.arange(K)])*np.log1p(-Th**2)).sum()
    #return lp
    # save the space when K is very large (e.g. K>100000)
    lp = -N*Th.sum()**2 + (Th**2).sum(axis=-1)

    #lp = N*np.log1p(-Th[:,np.newaxis]*Th).sum() - N*np.log1p(-Th**2).sum()
    lp += (Edges[2,:]*np.log(Th[:,Edges[0,:]]*Th[:,Edges[1,:]]) -Edges[2,:]*np.log1p(-Th[:,Edges[0,:]]*Th[:,Edges[1,:]])).sum(axis=-1)

    return lp



def log_like(Edges, Th, N):
    ## add up everything
    #lp = (X*np.log(Th) + np.log(Th)[:,np.newaxis]*X + (N-X)*np.log1p(-Th[:,np.newaxis]*Th)).sum()

    ## remove the diagonal
    #K = X.shape[0]
    #lp -= (X[np.arange(K), np.arange(K)]*2*np.log(Th) + (N-X[np.arange(K), np.arange(K)])*np.log1p(-Th**2)).sum()
    #return lp
    # save the space when K is very large (e.g. K>100000)
    lp = 2*N*np.array([np.log1p(-Th[:,i]*Th[:,i+1:]).sum(axis=-1) for i in range(len(Th.shape[1])-1)]).sum(axis=-1)
    #lp = N*np.log1p(-Th[:,np.newaxis]*Th).sum() - N*np.log1p(-Th**2).sum()
    lp += (Edges[2,:]*np.log(Th[:,Edges[0,:]]*Th[:,Edges[1,:]]) -Edges[2,:]*np.log1p(-Th[:,Edges[0,:]]*Th[:,Edges[1,:]])).sum(axis=-1)

    return lp

def log_like_thk(k, Edges, Th, N):
    lp = 2*N*np.log1p(-Th[:,k][:,np.newaxis]*Th).sum(axis=-1) - 2*N*np.log1p(-Th[:,k]**2)
    idcs = (Edges[0,:] == k) | (Edges[1,:] == k)
    lp += (Edges[2,idcs]*np.log(Th[:,Edges[0,idcs]]*Th[:,Edges[1,idcs]]) -Edges[2,idcs]*np.log1p(-Th[:,Edges[0,idcs]]*Th[:,Edges[1,idcs]])).sum(axis=-1)
    return lp

####################################

################
################
## joint prob
################
################

def log_prob(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)
    Th = constrain_rates(uTh)

    lalp = log_prior_alph(alph, alpha_a, alpha_b)

    lgam = log_prior_gam(gam, gamma_a, gamma_b)

    llam = log_prior_lamb(lamb, lambda_a, lambda_b)

    lth = log_prior_th(alph, gam, lamb, Th)

    #ll = log_like(Edges, Th, N)
    ll = approx_log_like(Edges, Th, N)

    ljac = log_jac(ualph, ugam, ulamb, uTh)

    return lalp + lgam + llam + lth + ll + ljac

def log_prob_alpha(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)
    Th = constrain_rates(uTh)

    lalp = log_prior_alph(alph, alpha_a, alpha_b)

    lth = log_prior_th(alph, gam, lamb, Th)

    ljac = log_jac_alph(ualph, ugam, ulamb, uTh)

    return lalp + lth + ljac

def log_prob_lambda(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)
    Th = constrain_rates(uTh)

    llam = log_prior_lamb(lamb, lambda_a, lambda_b)

    lth = log_prior_th(alph, gam, lamb, Th)

    ljac = log_jac_lamb(ualph, ugam, ulamb, uTh)

    return llam + lth + ljac

def log_prob_thk(k, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)
    #Th = constrain_rates(uTh)

    lth = log_prior_thk(k, alph, gam, lamb, Th)

    ll = log_like_thk(k, Edges, Th, N)

    ljac = log_jac_thk(k, ualph, ugam, ulamb, uTh)

    return lth + ll + ljac

def log_prob_thK(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)
    Th = constrain_rates(uTh)

    lth = log_prior_th(alph, gam, lamb, Th)

    ll = log_like_thk(uTh.shape[1]-1, Edges, Th, N)

    ljac = log_jac_thK(ualph, ugam, ulamb, uTh)

    return lth + ll + ljac

################
################
## sequential rep
################
################


def rej_beta(K, alph, gam, lamb, start=None):
    """
    using rejection representation to sample theta from the prior distribution nu(theta, alph)
    """
    S = alph.shape[0]
    Th = np.zeros((S,K))

    for s in range(S):
        g1 = gam[s]*np.exp(gammaln(lamb[s]+1)-gammaln(1-alph[s])-gammaln(lamb[s]+alph[s]))
        if start==None:
            g = 0
        elif alph[s]==0:
            g = -g1*np.log(start)
        elif alph[s]>0:
            g = g1/alph[s]*(np.exp(-alph[s]*np.log(start))-1)
        k = 0
        while k<K:
            g += np.random.exponential(1)
            lv = -g/g1 if alph[s]==0 else (-1/alph[s])*np.log(1+alph[s]*g/g1)
            u = np.random.uniform()
            if np.log(u)<=(lamb[s]+alph[s]-1)*np.log(1-np.exp(lv)):
                Th[s,k] = np.exp(lv)
                k += 1

    return Th

def rej_beta_multiple(K, alphs, gams, lambs, _Ths):
    g1s = gams*np.exp(gammaln(lambs+1)-gammaln(1.-alphs)-gammaln(lambs+alphs))
    zidcs = (alphs == 0)
    nzidcs = (alphs > 0)

    lastgs = np.zeros(alphs.shape[0])
    lastgs[zidcs] = -g1s[zidcs]*np.log(_Ths[zidcs, -1])
    lastgs[nzidcs] = g1s[nzidcs]/alphs[nzidcs]*(np.exp(-alphs[nzidcs]*np.log(_Ths[nzidcs, -1]))-1)

    _K = K+20
    gs = np.zeros((alphs.shape[0], _K))
    Ths = np.zeros((_Ths.shape[0], K+_K))
    Ths[:, :_Ths.shape[1]] = _Ths
    while True:
        gs[:, :] = lastgs[:, np.newaxis]
        gs += np.random.exponential(1., size=(alphs.shape[0], _K)).cumsum(axis=1)

        lastgs = gs[:, -1].copy()

        gs[zidcs, :] = -gs[zidcs, :] / g1s[zidcs][:,np.newaxis]
        gs[nzidcs,:] = (-1./alphs[nzidcs])[:,np.newaxis]*np.log(1.+ (alphs[nzidcs]/g1s[nzidcs])[:,np.newaxis]*gs[nzidcs,:])
        gs = np.exp(gs)
        us = np.random.rand(gs.shape[0], gs.shape[1])
        acs = (np.log(us) <= (np.log1p(-gs)* (lambs+alphs-1.)[:,np.newaxis]))
        gs[~acs] = 0.

        Ths[:, -_K:] = gs
        Ths.sort(axis=1)
        Ths = Ths[:, ::-1]
        if np.all(Ths[:, :K] > 0.):
            return Ths[:, :K]

################
################
## sampling moves
################
################


def gibbs_gamma(S, K, alph, gam, lamb, Th, gamma_a, gamma_b):
    ap = gamma_a + K
    bp = gamma_b + nu_int(Th[:,-1], alph, gam, lamb)/gam
    return np.random.gamma(ap, 1./bp, size = S)

def mh_alpha(S, alpha_step, ualph, ualph_lower, ualph_upper, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    # transform into reals
    #print('alpha: ' + str(np.exp(ualph)/(1+np.exp(ualph))))
    #print('ualpha: ' + str(ualph))
    ualph_prime = unconstrain_alpha(ualph, ualph_lower, ualph_upper)
    #print('ualpha prime: ' + str(ualph_prime))
    #print()
    # proposal in unconstrained space
    ualph_primep = ualph_prime + alpha_step*np.random.randn(S)
    #print('proposal: ' + str(ualph_primep))
    # return proposal to constrained space
    ualphp = constrain_alpha(ualph_primep, ualph_lower, ualph_upper)
    #print('proposal in ualpha space: ' + str(ualphp))
    #print()
    # calculate probabilities accounting for jacobian
    lp0 = log_prob_alpha(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    #print('lp0 w/o jacobian: ' + str(lp0))
    lp0 += log_jac_ualph(ualph_prime, ualph_lower, ualph_upper)  # jacobian
    #print('log jacobian: ' + str(log_jac_ualph(ualph_prime, ualph_lower, ualph_upper)))
    #print('lp0: ' + str(lp0))
    #print()
    lp1 = log_prob_alpha(ualphp, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    #print('lp1 w/o jacobian: ' + str(lp1))
    lp1 += log_jac_ualph(ualph_primep, ualph_lower, ualph_upper) # jacobian
    #print('log jacobian: ' + str(log_jac_ualph(ualph_primep, ualph_lower, ualph_upper)))
    #print('lp1: ' + str(lp1))

    out = ualph
    swaps = np.log(np.random.rand(S)) <= lp1 - lp0
    out[swaps] = ualphp[swaps]

    #if np.log(np.random.rand(S)) <= lp1 - lp0:
    #    return ualphp
    #return ualph

    return out

def mh_lambda(S, lambda_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    ulambp = ulamb + lambda_step*np.random.randn(S)
    lp0 = log_prob_lambda(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    lp1 = log_prob_lambda(ualph, ugam, ulambp, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)

    out = ulamb
    swaps = np.log(np.random.rand(S)) <= lp1 - lp0
    out[swaps] = ulambp[swaps]
    #if np.log(np.random.rand(S)) <= lp1 - lp0:
    #    return ulambp
    #return ulamb
    return out

def mh_uThmk(S, k, Th_step, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    uTh_old = uTh[:,k]
    Th_old = Th[:,k]
    if k < uTh.shape[1]-1:
        lp0 = log_prob_thk(k, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        uTh[:,k] += Th_step*np.random.randn(S)
        Th[:,k] = constrain_rate_k(k, uTh)
        lp1 = log_prob_thk(k, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    else:
        lp0 = log_prob_thK(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        uTh[:,k] += Th_step*np.random.randn(S)
        lp1 = log_prob_thK(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)

    out1 = uTh_old
    out2 = Th_old
    swaps = np.log(np.random.rand(S)) <= lp1 - lp0
    out1[swaps] = uTh[swaps,k]
    out2[swaps] = Th[swaps,k]

    #if np.log(np.random.rand(S)) <= lp1 - lp0:
    #    return uTh[:,k], Th[:,k]
    #return uTh_old, Th_old

    return out1, out2

def mh_uTh0(S, idx0, Th_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    uTh_old = uTh[:,idx0]
    lp0 = log_prob(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    uTh[:,idx0] += Th_step*np.random.randn(S, len(idx0))
    lp1 = log_prob(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)

    out = uTh
    swaps = np.log(np.random.rand(S)) <= lp1 - lp0
    out[swaps,:] = uTh[swaps,:]
    out[~swaps,:][:,idx0] = uTh_old[~swaps,:]


    #if np.log(np.random.rand(S)) <= lp1 - lp0:
    #    return uTh
    #uTh[:,idx0] = uTh_old
    #return uTh

    return out


#################
#################
## data
#################
#################

# load the network
df = pd.read_csv('../fb2.txt', delim_whitespace=True, )

# convert the datetime to integers
df['rnd'] = pd.to_datetime(df['date']).astype(int)/(1800*10**9) # this converts to 1/2 hour intervals
df['rnd'] = (df['rnd'] - df['rnd'][0]).astype(int) #this shifts the intervals by the first time

# remove any self loop entries
df = df[df.v1 != df.v2]

# shift the indices down by 1 (start at 0)
df.v1 -= 1
df.v2 -= 1


# convert the relevant columns to numpy arrays
v1 = np.array(df.v1)
v2 = np.array(df.v2)
rnds = np.array(df.rnd)

# threshold the rounds

rnd_begin = 3950
rnd_end = 10377

v1 = v1[(rnds >= rnd_begin) & (rnds < rnd_end)]
v2 = v2[(rnds >= rnd_begin) & (rnds < rnd_end)]
rnds = rnds[(rnds >= rnd_begin) & (rnds < rnd_end)]


# collect the number of edges and vertices binned into days
# censor to the range when the first edge appears, and before a shift in rate happens
nE = [0] # because we use nE[-1] below; we will remove this entry right after
nV = []
nN = []
V = set()
for n in range(rnd_begin, rnd_end):
    idcs = np.where(rnds == n)[0]
    nE.append(nE[-1] + idcs.shape[0])
    for i in idcs:
        V.add(v1[i])
        V.add(v2[i])
    nV.append(len(V))
    nN.append(n+1)
nE = np.array(nE)[1:] #remove the extra first entry we put before
nV = np.array(nV)
nN = np.array(nN)

# collect the edges into a sparse representation
N = rnd_end - rnd_begin
K = max(v1.max(), v2.max())+1
X = np.zeros((K,K))
for i in range(v1.shape[0]):
    X[v1[i], v2[i]] += 1
degrees = (X+X.T).sum(axis=0)
hist, edges = np.histogram(degrees, density=True, bins=50)
edges = edges[1:]

idcs = np.where(X>0)
Obs = np.zeros((3, idcs[0].shape[0]), dtype=np.int64)
Obs[0,:] = idcs[0]
Obs[1,:] = idcs[1]
Obs[2,:] = X[idcs[0], idcs[1]]

#################
#################
## main samplers
#################
#################

# this is the actual sampler
def adaptive_sampler(K, T, S = 1, alph = 0.5, gam = 2., lamb = 20., Th = None, verbose = False):
    """
    K : truncation size
    T : number of steps to run the chain for
    S : number of samples to generate
    alph, gam, lamb, Th: initial values of alpha, gamma, lambda, Thetas
        alph, gam, lamb are 1d arrays and Th is shape(1,K)
    verbose : boolean indicating whether to print messages

    returns three shape(S,1) arrays with samples of alpha, gamma, and lambda (resp)
    and a shape(S,K) array with samples of Th
    """
    global Obs

    # sampler settings (these don't change)
    Edges = np.copy(Obs)
    # GC hack to truncate Edges
    idx = np.logical_and(Edges[0,:] < K, Edges[1,:] < K)
    Edges = Edges[:,idx]
    #gam = 2.
    #lamb = 20.
    # prior distribution params
    alpha_a = 0.5
    alpha_b = 1.5
    gamma_a = 1.
    gamma_b = 1.
    lambda_a = 1.
    lambda_b = 1.
    # steps
    lambda_step = 0.1
    Th0_step = 0.03
    Th1_step = 0.1
    ThK_step = 0.1
    #Th0 = None

    #np.seterr(all='raise')

    # make sure the edges array only contains pairs of indices with edge count > 0
    Edges = Edges[:, Edges[2,:]>0]

    # extract the set of nonempty vertices
    idx1 = np.unique(Edges[:-1, :]).astype(int)
    # GC hack to create only vertices with k < K
    idx1 = idx1[idx1 <= K]

    # extract empty verts
    idx0 = []
    for k in range(K):
        if k not in idx1:
            idx0.append(k)
    idx0 = np.array(idx0)


    # storage for samples
    alphs = None
    lambs = None
    gams = None
    Ths = None
    # start sampling
    if verbose: print("Sampling start; K = " + str(K))
    alph = alph*np.ones(S)
    lamb = lamb*np.ones(S)
    gam = gam*np.ones(S)

    # initialize the rates from the prior if not specified
    if Th is None:
        Th = rej_beta(K, alph, gam, lamb)
    else:
        Th = Th.reshape(1,K)*np.ones((S,K))


    # compute alpha step and limits
    alpha_step = alpha_stepsize(alph)
    ualph_lower, ualph_upper = alpha_lims(alph)

    # compute the unconstrained versions of them
    ualph, ugam, ulamb = unconstrain_hyper(alph, gam, lamb)
    uTh = unconstrain_rates(Th)

    #alph_accept = 1
    #lamb_accept = 1
    #th0_accept = 1
    #th1_accept = 1
    #thK_accept = 1
    for i in np.arange(T):
        #print('\nadaptive sampler iter ' + str(i+1) + '/' + str(T), end='\r')
        if i%50==0 and verbose:
            print('iter: ' + str(i))
            print('alphas: ' + str(alph))
            print('gammas: ' + str(gam))
            print('lambdas: ' + str(lamb))
            print()

        # Basic MH for alpha, lambda, gibbs for gamma
        gam = gibbs_gamma(S, K, alph, gam, lamb, Th, gamma_a, gamma_b)
        ualph, ugam, ulamb = unconstrain_hyper(alph, gam, lamb)

        ulamb = mh_lambda(S, lambda_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)

        ualph = mh_alpha(S, alpha_step, ualph, ualph_lower, ualph_upper, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)

        # individual MH steps for idx1 entries
        for k in idx1: uTh[:,k], Th[:,k] = mh_uThmk(S, k, Th1_step, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)

        # joint move for idx0 entries
        uTh = mh_uTh0(S, idx0, Th0_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        Th = constrain_rates(uTh)

        # individual MH step for K entry
        uTh[:,-1], Th[:,-1] = mh_uThmk(S, uTh.shape[1]-1, ThK_step, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        Th = constrain_rates(uTh)

        #store the results
        alphs = alph
        gams = gam
        lambs = lamb
        Ths = Th

    return alphs, gams, lambs, Ths





def sampler_wrapper(y, T, S, logp):
    # this function simulates NxS for from the adaptive sampler chains T steps
    #
    # y    : shape(N,K) array of params
    #        y[:,0] alphas, y[:,1] gammas, y[:,2] lambdas, y[:,3:] thetas
    # T    : shape(N,) array with no. steps per location
    # S    : sample size
    # logp : target logdensity (ignored but required by lbvi routines)
    #
    # out : array of shape(S, N, K)

    N = y.shape[0] # should be 7, i.e. the number of bins in alpha
    K = y.shape[1] # should be 2013, i.e. 3 main params + 2010 thetas
    out = np.zeros((S,N,K))

    for n in range(N):
        #print('\nsampler wrapper kernel ' + str(n+1) + '/' + str(N), end='\r')
        # obtain samples
        Alphs, Gams, Lambs, Ths = adaptive_sampler(T = T[n], S = S, alph = y[n,0], gam = y[n,1], lamb = y[n,2], Th = y[n,3:])
        # save in array
        out[:,n,0] = Alphs
        out[:,n,1] = Gams
        out[:,n,2] = Lambs
        out[:,n,3:] = Ths
    # end for

    return out



def kernel_sampler(y, T, S, logp, t_increment, chains = None, update = False):
    # this function determines the incremental runs needed and calls the above gaussian sampler
    #
    # shape(N,K) array of locations y
    # shape(N,) array with no. steps per location T
    # sample size S
    # target logdensity logp
    # t_increment is the incremental steps taken at each iteration per kernel
    # chains is a list of N shape(T_n,K) arrays with the current chains
    #   it is used for cacheing. If None, then the whole chain is generated from scratch
    # update is boolean. If True and chains is not None, a new chains list will be created with the generated samples appended
    #
    # out: array of shape(S, N, K)

    # if chain has to be run from scratch, call gaussian sampler with full T
    if chains is None:
        #print('sampling from chains')
        #print('sample: ' + str(y))
        #print('steps: ' + str(T))
        #print('sample size: ' + str(S))
        return sampler_wrapper(y, T, S, logp)
    else:
        # determine the current T in the chains
        N = y.shape[0]
        K = y.shape[1]
        Tnew = np.zeros(N)
        ynew = np.zeros((N,K))
        #print('chains: ' + str(chains))
        for n in range(N):
            # determine incremental T's
            Tcurr = t_increment*(chains[n].shape[0] - 1) # -1 to account for initial sample
            Tnew[n] = T[n] - Tcurr # chains[n] is shape(T_n,K)
            if Tnew[n] < 0:
                print('error! negative incremental steps')
                print('n = ' + str(n))
                print('current length: ' + str(Tcurr))
                print('needed length: ' + str(T[n]))
                print('incremental steps: ' + str(Tnew[n]))

            # update starting points to the previous iter (to account for +1)
            ynew[n,:] = chains[n][chains[n].shape[0]-1,:]
        # end for

        # now call gaussian sampler with the incremental samples only
        #print('sampling ' + str(Tnew) + ' instead of ' + str(T))
        #print('sampling from chains')
        #print('sample: ' + str(ynew))
        #print('steps: ' + str(Tnew+1))
        #print('sample size: ' + str(S))
        samples = sampler_wrapper(ynew, Tnew+1, S, logp) #(S,N,K)

        if update:
            # append first sample to each chain
            for n in range(N):
                if Tnew[n] > 0: chains[n] = np.vstack((chains[n], samples[0,n,:].reshape(1,K)))
            # end for

            return samples, chains
        else:
            return samples
