import autograd.numpy as np
#from autograd.scipy.special import gammaln, digamma, polygamma, betaln, betainc, erf, erfinv
from autograd.scipy.special import gammaln, digamma, polygamma, betaln, erf, erfinv
from autograd_gamma import betainc
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
## nu_int
################
################
def nu_int(x, alph, gam, lamb):

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
    term2 = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1] - lmax)))
    #Th[:,-1] = np.exp(-lmax - np.log(np.exp(0-lmax) + np.exp(-uTh[:,-1] - lmax)))

    # get Th[1...K-1]
    # AUTOGRAD FIX
    #tmp = np.zeros((3, uTh.shape[0], uTh.shape[1]-1))
    #tmp[1, :,:] = -uTh[:,:-1]
    #tmp[2, :,:] = -uTh[:,-1][:,np.newaxis]
    tmp_term1 = -uTh[:,:-1][np.newaxis,:,:]
    tmp_term2 = -uTh[:,-1][np.newaxis,:,np.newaxis]*np.ones(uTh.shape[1]-1)
    tmp =  np.concatenate((np.zeros((1,uTh.shape[0], uTh.shape[1]-1)), tmp_term1, tmp_term2), axis=0)

    lmax = tmp.max(axis=0)
    lnum = lmax + np.log( np.exp(tmp - lmax).sum(axis=0) )
    lmax = tmp[:2,:,:].max(axis=0)
    ldenom = lmax + np.log( np.exp(tmp[:2,:,:] - lmax).sum(axis=0) )
    term1 = np.exp( lnum - ldenom + np.log(Th[:,-1][:,np.newaxis]) )
    # AUTOGRAD FIX
    #Th[:,:-1] = np.exp( lnum - ldenom + np.log(Th[:,-1][:,np.newaxis]) )
    Th = np.hstack((term1, term2.reshape(uTh.shape[0],1)))
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
    # AUTOGRAD FIX
    #uTh = np.zeros((Th.shape[0],Th.shape[1]))
    #uTh[:,-1] = np.log(Th[:,-1]) - np.log1p(-Th[:,-1])
    #uTh[:,:-1] = np.log(Th[:,:-1]) + np.log1p(-(Th[:,-1])[:,np.newaxis]/Th[:,:-1]) - np.log1p(-Th[:,:-1])

    # the above (original) doesn't play well with autograd; the code below does the same but works with autograd
    tmp1 = np.log(Th[:,:-1]) + np.log1p(-(Th[:,-1])[:,np.newaxis]/Th[:,:-1]) - np.log1p(-Th[:,:-1])
    tmp2 = np.log(Th[:,-1]) - np.log1p(-Th[:,-1])
    uTh = np.hstack((tmp1, tmp2.reshape(Th.shape[0],1)))
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
    # AUTOGRAD FIX
    #tmp = np.zeros((2, uTh.shape[0], uTh.shape[1]-1))
    #tmp[1, :,:] = -uTh[:,:-1]
    tmp_term1 = -uTh[:,:-1][np.newaxis,:,:]
    tmp = np.vstack((np.zeros((1, uTh.shape[0], uTh.shape[1]-1)), tmp_term1))
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

    ll = log_like_thk(uTh.shape[0]-1, Edges, Th, N)

    ljac = log_jac_thK(ualph, ugam, ulamb, uTh)

    return lth + ll + ljac


#################
#################
## data
#################
#################

# load the network
df = pd.read_csv('../network-model/fb2.txt', delim_whitespace=True, )

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
NN = rnd_end - rnd_begin
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



################
################
## relevant functions
################
################


def logp_aux(x, K = 2013):
    # x    : shape(N,K) array of params
    #        x[:,0] alphas, x[:,1] gammas, x[:,2] lambdas, x[:,3:] thetas

    # load data
    global Obs
    Edges = np.copy(Obs)

    # retrieve param values
    alph = x[:,0]
    gam = x[:,1]
    lamb = x[:,2]
    Th = x[:,3:]

    # compute the unconstrained versions of them
    ualph, ugam, ulamb = unconstrain_hyper(alph, gam, lamb)
    uTh = unconstrain_rates(Th)

    # prior distribution params
    alpha_a = 0.5
    alpha_b = 1.5
    gamma_a = 1.
    gamma_b = 1.
    lambda_a = 1.
    lambda_b = 1.

    return log_prob(ualph, ugam, ulamb, uTh, Edges, NN, alpha_a, alpha_b, gamma_a, gamma_b, lambda_a, lambda_b)


def sample(size, K):
    # instead of generating a sample, we load the one we generated before
    # size and K are ignored; the sample was already generated with desired size,K values anyway
    return np.load('../network-model/initial_sample.npy')


# CREATE WEIGHT OPT SCHEDULE AND MAXITER
def w_maxiters(k, long_opt = False):
    if k > 10: return 10
    if k == 0: return 10
    if long_opt: return 10
    return 10


def w_schedule(k):
    #if k == 0: return 0.1
    return 1.
