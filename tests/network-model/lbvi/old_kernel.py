import numpy as np
from scipy.special import gammaln, digamma, polygamma, betaln, betainc, erf, erfinv
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
    # given alpha in (0,1), return mh sampler limits in logit (unconstrained) space
    if alpha < 0.02:
        lower = np.NINF
        upper = np.log(0.02) - np.log1p(-0.02)
    elif alpha < 0.1:
        lower = np.log(0.02) - np.log1p(-0.02)
        upper = np.log(0.1) - np.log1p(-0.1)
    elif alpha < 0.2:
        lower = np.log(0.1) - np.log1p(-0.1)
        upper = np.log(0.2) - np.log1p(-0.2)
    elif alpha < 0.4:
        lower = np.log(0.2) - np.log1p(-0.2)
        upper = np.log(0.4) - np.log1p(-0.4)
    elif alpha < 0.6:
        lower = np.log(0.4) - np.log1p(-0.4)
        upper = np.log(0.6) - np.log1p(-0.6)
    elif alpha < 0.8:
        lower = np.log(0.6) - np.log1p(-0.6)
        upper = np.log(0.8) - np.log1p(-0.8)
    else:
        lower = np.log(0.8) - np.log1p(-0.8)
        upper = np.inf
    return lower, upper

def alpha_stepsize(alpha):
    # given alpha in (0,1), return mh sampler step size
    if alpha < 0.02:
        step = 0.5
    #elif alpha < 0.1:
    #    step = 0.1
    #elif alpha < 0.2:
    #    step = 0.04
    #elif alpha < 0.4:
    #    step = 0.04
    #elif alpha < 0.6:
    #    step = 0.04
    #elif alpha < 0.8:
    #    step = 0.04
    else:
        step = 0.1
    return step


def constrain_alpha(ualph_prime, ualph_lower, ualph_upper):
    if ualph_lower < -100: # ==-np.NINF
        return ualph_upper - np.exp(ualph_prime)
    elif ualph_upper > 100: # ==np.inf
        return ualph_lower + np.exp(ualph_prime)
    else:
        return ualph_lower + (ualph_upper - ualph_lower)*np.exp(ualph_prime)/(1+np.exp(ualph_prime))

def unconstrain_alpha(ualph, ualph_lower, ualph_upper):
    if ualph_lower < -100: # ==-np.NINF
        return np.log(ualph_upper - ualph)
    elif ualph_upper > 100: # ==np.inf
        return np.log(ualph - ualph_lower)
    else:
        return np.log((ualph - ualph_lower) / (ualph_upper - ualph))


def log_jac_ualph(ualph_prime, ualph_lower, ualph_upper):
    if ualph_lower < -100 or ualph_upper > 100: # ==-np.NINF or ==np.inf
        return ualph_prime
    else:
        return np.log(np.abs(ualph_lower + (ualph_upper - ualph_lower)*np.exp(ualph_prime)/(1+np.exp(ualph_prime))**2))

################
################
## nu_int
################
################
def nu_int(x, alph, gam, lamb):
    if alph==0:
        if np.floor(lamb) != lamb:
            raise ValueError("lamb must be an integer for this method to work")
        lamb = int(np.floor(lamb))
        lls = np.zeros(lamb)
        for n in range(lamb-1):
             lls[n] = gammaln(lamb) - gammaln(n+1) - gammaln(lamb-n) - np.log(lamb-n-1.) + np.log(1.-x**(lamb-n-1.)) + np.log(gam) + np.log(lamb)
        lls[lamb-1] = np.log(gam) + np.log(lamb) + np.log(-np.log(x))
        llmax = lls.max()
        lls -= llmax
        return np.exp(llmax + np.log((np.exp(lls)*(-1)**(lamb - 1 - np.arange(lamb))).sum()))
    else:
        ll1 = -alph*np.log(x) + (lamb+alph-1.)*np.log1p(-x) - np.log(lamb+alph-1.) - betaln(1.-alph, lamb+alph-1.)
        ll2 = np.log(betainc(1.-alph, lamb+alph-1., x))
        llmax = max(ll1, ll2)
        ll1 -= llmax
        ll2 -= llmax
        ll = llmax + np.log(np.exp(ll1) + np.exp(ll2))
        if ll<0:
            print("ll1: ", ll1, "ll2: ", ll2, "llmax: ", llmax, "theta_K: ", x )
        m1 = 0
        llmax = max(m1, ll)
        ll -= llmax
        m1 -= llmax
        ll = llmax + np.log(np.exp(ll) - np.exp(m1))
        ll += np.log(gam) + np.log(lamb) - np.log(alph)
        return np.exp(ll)

################
################
## transforms
################
################

def constrain_hyper(ualph, ugam, ulamb):
    # get alpha
    lmax = max(0, -ualph)
    alph = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-ualph - lmax)))
    # get gamma
    gam = np.exp(ugam)
    # get lambda
    lamb = 1+np.exp(ulamb)
    return alph, gam, lamb

def constrain_rates(uTh):
    # allocate memory for Thetas
    Th = np.zeros(uTh.shape[0])

    # get Theta[K]
    lmax = max(0, -uTh[-1])
    Th[-1] = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-uTh[-1] - lmax)))

    # get Th[1...K-1]
    tmp = np.zeros((3, uTh.shape[0]-1))
    tmp[1, :] = -uTh[:-1]
    tmp[2, :] = -uTh[-1]
    lmax = tmp.max(axis=0)
    lnum = lmax + np.log( np.exp(tmp - lmax).sum(axis=0) )
    lmax = tmp[:2,:].max(axis=0)
    ldenom = lmax + np.log( np.exp(tmp[:2,:] - lmax).sum(axis=0) )
    Th[:-1] = np.exp( lnum - ldenom + np.log(Th[-1]) )
    return Th

def constrain_rate_k(k, uTh):
    # get Theta[K]
    lmax = max(0, -uTh[-1])
    ThK = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-uTh[-1] - lmax)))

    # get Th[1...K-1]
    tmp = np.zeros((3, 1))
    tmp[1, 0] = -uTh[k]
    tmp[2, 0] = -uTh[-1]
    lmax = tmp.max(axis=0)
    lnum = lmax + np.log( np.exp(tmp - lmax).sum(axis=0) )
    lmax = tmp[:2,0].max(axis=0)
    ldenom = lmax + np.log( np.exp(tmp[:2,0] - lmax).sum(axis=0) )
    return np.exp( lnum - ldenom + np.log(ThK) )

def unconstrain_hyper(alph, gam, lamb):
    ualph = np.log(alph) - np.log1p(-alph)
    ugam = np.log(gam)
    ulamb = np.log(lamb-1.)
    return ualph, ugam, ulamb

def unconstrain_rates(Th):
    uTh = np.zeros(Th.shape[0])
    uTh[-1] = np.log(Th[-1]) - np.log1p(-Th[-1])
    uTh[:-1] = np.log(Th[:-1]) + np.log1p(-Th[-1]/Th[:-1]) - np.log1p(-Th[:-1])
    return uTh



##############################
##############################
##############################
##############################

def log_jac(ualph, ugam, ulamb, uTh):
    # alpha vs ualph jac
    lmax = max(0, -ualph)
    logjac_alph = -ualph  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-ualph-lmax)))

    # gam vs ugam jac
    logjac_gam = ugam

    # lamb vs ulamb jac
    logjac_lamb = ulamb

    # Th[-1] vs uTh[-1] jac
    lmax = max(0, -uTh[-1])
    logjac_thK = -uTh[-1]  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[-1]-lmax)))

    # this just makes it e^{-tK}/()^1 instead of ()^2
    logjac_thk = logjac_thK + (lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[-1]-lmax)))

    # Th[1...K-1] vs t[1...K-1] jac
    tmp = np.zeros((2, uTh.shape[0]-1))
    tmp[1, :] = -uTh[:-1]
    lmax = tmp.max(axis=0)
    logjac_thk = logjac_thk -uTh[:-1] - 2*(lmax + np.log(np.exp(tmp-lmax).sum(axis=0)) )

    # the jacobian matrix is lower triangular, so can just add these up and return
    return logjac_alph + logjac_gam + logjac_lamb + logjac_thK + logjac_thk.sum()


def log_jac_alph(ualph, ugam, ulamb, uTh):
    # alpha vs ualph jac
    lmax = max(0, -ualph)
    logjac_alph = -ualph  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-ualph-lmax)))
    return logjac_alph

def log_jac_lamb(ualph, ugam, ulamb, uTh):
    # lamb vs ulamb jac
    logjac_lamb = ulamb
    return logjac_lamb


def log_jac_thk(k, ualph, ugam, ulamb, uTh):
    uThK = uTh[-1]
    lmax = max(0, -uThK)
    logjac_thK = -uThK  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uThK-lmax)))

    # this just makes it e^{-tK}/()^1 instead of ()^2
    logjac_thk = logjac_thK + (lmax + np.log(np.exp(0-lmax) + np.exp(-uThK-lmax)))

    # Th[1...K-1] vs t[1...K-1] jac
    uThk = uTh[k]
    lmax = max(0, -uThk)
    logjac_thk = logjac_thk -uThk - 2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uThk-lmax)))
    return logjac_thK + logjac_thk

def log_jac_thK(ualph, ugam, ulamb, uTh):
    # Th[-1] vs uTh[-1] jac
    lmax = max(0, -uTh[-1])
    logjac_thK = -uTh[-1]  -2*(lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[-1]-lmax)))

    # this just makes it e^{-tK}/()^1 instead of ()^2
    logjac_thk = logjac_thK + (lmax + np.log(np.exp(0-lmax) + np.exp(-uTh[-1]-lmax)))

    # Th[1...K-1] vs t[1...K-1] jac
    tmp = np.zeros((2, uTh.shape[0]-1))
    tmp[1, :] = -uTh[:-1]
    lmax = tmp.max(axis=0)
    logjac_thk = logjac_thk -uTh[:-1] - 2*(lmax + np.log(np.exp(tmp-lmax).sum(axis=0)) )

    # the jacobian matrix is lower triangular, so can just add these up and return
    return logjac_thK + logjac_thk.sum()

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
    lp = (-1.-alph)*np.log(Th) + (lamb+alph-1.)*np.log1p(-Th)
    # normalizing constant
    lp += np.log(gam) + gammaln(lamb+1) - gammaln(lamb+alph) - gammaln(1-alph)
    # subtract nu_int at the end
    return lp.sum() - nu_int(Th[-1], alph, gam, lamb)

def log_prior_thk(k, alph, gam, lamb, Th):
    # the log beta process density
    lp = (-1.-alph)*np.log(Th[k]) + (lamb+alph-1.)*np.log1p(-Th[k])
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
    lp = -N*Th.sum()**2 + (Th**2).sum()

    #lp = N*np.log1p(-Th[:,np.newaxis]*Th).sum() - N*np.log1p(-Th**2).sum()
    lp += (Edges[2,:]*np.log(Th[Edges[0,:]]*Th[Edges[1,:]]) -Edges[2,:]*np.log1p(-Th[Edges[0,:]]*Th[Edges[1,:]])).sum()

    return lp



def log_like(Edges, Th, N):
    ## add up everything
    #lp = (X*np.log(Th) + np.log(Th)[:,np.newaxis]*X + (N-X)*np.log1p(-Th[:,np.newaxis]*Th)).sum()

    ## remove the diagonal
    #K = X.shape[0]
    #lp -= (X[np.arange(K), np.arange(K)]*2*np.log(Th) + (N-X[np.arange(K), np.arange(K)])*np.log1p(-Th**2)).sum()
    #return lp
    # save the space when K is very large (e.g. K>100000)
    lp = 2*N*np.array([np.log1p(-Th[i]*Th[i+1:]).sum() for i in range(len(Th)-1)]).sum()
    #lp = N*np.log1p(-Th[:,np.newaxis]*Th).sum() - N*np.log1p(-Th**2).sum()
    lp += (Edges[2,:]*np.log(Th[Edges[0,:]]*Th[Edges[1,:]]) -Edges[2,:]*np.log1p(-Th[Edges[0,:]]*Th[Edges[1,:]])).sum()

    return lp

def log_like_thk(k, Edges, Th, N):
    lp = 2*N*np.log1p(-Th[k]*Th).sum() - 2*N*np.log1p(-Th[k]**2)
    idcs = (Edges[0,:] == k) | (Edges[1,:] == k)
    lp += (Edges[2,idcs]*np.log(Th[Edges[0,idcs]]*Th[Edges[1,idcs]]) -Edges[2,idcs]*np.log1p(-Th[Edges[0,idcs]]*Th[Edges[1,idcs]])).sum()
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

    ll = log_like_thk(uTh.shape[0]-1, Edges, Th, N)

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
    g1 = gam*np.exp(gammaln(lamb+1)-gammaln(1-alph)-gammaln(lamb+alph))
    Th = np.zeros(K)
    if start==None:
        g = 0
    elif alph==0:
        g = -g1*np.log(start)
    elif alph>0:
        g = g1/alph*(np.exp(-alph*np.log(start))-1)

    k = 0
    while k<K:
        g += np.random.exponential(1)
        lv = -g/g1 if alph==0 else (-1/alph)*np.log(1+alph*g/g1)
        u = np.random.uniform(0, 1)
        if np.log(u)<=(lamb+alph-1)*np.log(1-np.exp(lv)):
            Th[k] = np.exp(lv)
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


def gibbs_gamma(K, alph, gam, lamb, Th, gamma_a, gamma_b):
    ap = gamma_a + K
    bp = gamma_b + nu_int(Th[-1], alph, gam, lamb)/gam
    return np.random.gamma(ap, 1./bp)

def mh_alpha(alpha_step, ualph, ualph_lower, ualph_upper, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    # transform into reals
    #print('alpha: ' + str(np.exp(ualph)/(1+np.exp(ualph))))
    #print('ualpha: ' + str(ualph))
    ualph_prime = unconstrain_alpha(ualph, ualph_lower, ualph_upper)
    #print('ualpha prime: ' + str(ualph_prime))
    #print()
    # proposal in unconstrained space
    ualph_primep = ualph_prime + alpha_step*np.random.randn()
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
    if np.log(np.random.rand()) <= lp1 - lp0:
        #print('accepted')
        return 1, ualphp
    #print()
    return 0, ualph

def mh_lambda(lambda_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    ulambp = ulamb + lambda_step*np.random.randn()
    lp0 = log_prob_lambda(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    lp1 = log_prob_lambda(ualph, ugam, ulambp, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    if np.log(np.random.rand()) <= lp1 - lp0:
        return 1, ulambp
    return 0, ulamb

def mh_uThmk(k, Th_step, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    uTh_old = uTh[k]
    Th_old = Th[k]
    if k < uTh.shape[0]-1:
        lp0 = log_prob_thk(k, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        uTh[k] += Th_step*np.random.randn()
        Th[k] = constrain_rate_k(k, uTh)
        lp1 = log_prob_thk(k, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    else:
        lp0 = log_prob_thK(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        uTh[k] += Th_step*np.random.randn()
        lp1 = log_prob_thK(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    if np.log(np.random.rand()) <= lp1 - lp0:
        return 1, uTh[k], Th[k]
    return 0, uTh_old, Th_old

def mh_uTh0(idx0, Th_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b):
    uTh_old = uTh[idx0]
    lp0 = log_prob(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    uTh[idx0] += Th_step*np.random.randn(len(idx0))
    lp1 = log_prob(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
    if np.log(np.random.rand()) <= lp1 - lp0:
        return 1, uTh
    uTh[idx0] = uTh_old
    return 0, uTh



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
## main sampler
#################
#################

def adaptive_sampler(T, alph = 0.5, gam = 2., lamb = 20., Th = None):
    """
    T     : number of samples to generate
    alph, gam, lamb, Th: initial values of alpha, gamma, lambda, Thetas
    """
    global Obs

    # sampler settings (these don't change)
    Edges = np.copy(Obs)
    K = 2010
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
    alpha_step = alpha_stepsize(alph)
    Th0_step = 0.03
    Th1_step = 0.1
    ThK_step = 0.1
    #Th0 = None

    #np.seterr(all='raise')

    # make sure the edges array only contains pairs of indices with edge count > 0
    Edges = Edges[:, Edges[2,:]>0]

    # extract the set of nonempty vertices
    idx1 = np.unique(Edges[:-1, :]).astype(int)

    # extraact empty verts
    idx0 = []
    for k in range(K):
        if k not in idx1:
            idx0.append(k)
    idx0 = np.array(idx0)

    # initialize the rates from the prior if not specified
    if Th is None: Th = rej_beta(K, alph, gam, lamb)

    # compute alpha limits
    ualph_lower, ualph_upper = alpha_lims(alph)

    # compute the unconstrained versions of them
    ualph, ugam, ulamb = unconstrain_hyper(alph, gam, lamb)
    uTh = unconstrain_rates(Th)

    # storage for samples
    alphs = None
    lambs = None
    gams = None
    Ths = None
    # start sampling
    print("Sampling start; K = " + str(K))
    burn = int(T/2)
    alphs = np.zeros(T-burn)
    lambs = np.zeros(T-burn)
    gams = np.zeros(T-burn)
    Ths = np.zeros((T-burn, K))

    alph_accept = 1
    lamb_accept = 1
    th0_accept = 1
    th1_accept = 1
    thK_accept = 1
    for i in np.arange(T):
        if i%50==0:
            lp = log_prob(ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
            print("i: {0:<5}".format(i),
                  "alph: {0:<5}".format(np.round(alph, 3)),
                  "alph_accept: {0:<5}".format(np.round(alph_accept/(i+1), 3)),
                  "gam: {0:<5}".format(np.round(gam, 3)),
                  "lamb: {0:<5}".format(np.round(lamb, 3)),
                  "lamb_accept: {0:<5}".format(np.round(lamb_accept/(i+1), 3)),
                  "log10Th_K: {0:<10}".format(np.log10(Th[-1]).round(3)),
                  "th0_accept: {0:<5}".format(np.round(th0_accept/(i+1), 3)),
                  "th1_accept: {0:<5}".format(np.round(th1_accept/(i+1), 3)),
                  "thK_accept: {0:<5}".format(np.round(thK_accept/(i+1), 3)),
                  "lp: {0:<10}".format(lp.round(3)))

        # Basic MH for alpha, lambda, gibbs for gamma
        gam = gibbs_gamma(K, alph, gam, lamb, Th, gamma_a, gamma_b)
        ualph, ugam, ulamb = unconstrain_hyper(alph, gam, lamb)

        _lamb_accept, ulamb = mh_lambda(lambda_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        lamb_accept += _lamb_accept
        alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)

        _alph_accept, ualph = mh_alpha(alpha_step, ualph, ualph_lower, ualph_upper, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        alph_accept += _alph_accept
        alph, gam, lamb = constrain_hyper(ualph, ugam, ulamb)

        # individual MH steps for idx1 entries
        _th1_accept = 0
        for k in idx1:
            __th1_accept, uTh[k], Th[k] = mh_uThmk(k, Th1_step, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
            _th1_accept += __th1_accept
        th1_accept += _th1_accept/len(idx1)

        # joint move for idx0 entries
        _th0_accept, uTh = mh_uTh0(idx0, Th0_step, ualph, ugam, ulamb, uTh, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        th0_accept += _th0_accept
        Th = constrain_rates(uTh)

        # individual MH step for K entry
        _thK_accept, uTh[-1], Th[-1] = mh_uThmk(uTh.shape[0]-1, ThK_step, ualph, ugam, ulamb, uTh, Th, Edges, N, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)
        thK_accept += _thK_accept
        Th = constrain_rates(uTh)

        #store the results
        if i >= burn:
            alphs[i-burn] = alph
            gams[i-burn] = gam
            lambs[i-burn] = lamb
            Ths[i-burn,:] = Th

    return alphs, gams, lambs, Ths
