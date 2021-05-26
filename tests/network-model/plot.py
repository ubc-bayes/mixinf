# plot the results of the network results

# PREAMBLE ####
import glob
#import autograd.numpy as np
#from autograd import elementwise_grad as egrad
import numpy as npc
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})
import argparse
import sys, os
import imageio
from sample_generation.sampler import adaptive_truncation_sampler

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../../lbvi/'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import lbvi # functions to do locally-adapted boosting variational inference
import numpy as np

# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="plot comparison between lbvi and other vi and mcmc routines")

parser.add_argument('--inpath', type = str, default = 'results/',
help = 'path of folder where csv files are stored')
parser.add_argument('--outpath', type = str, default = 'results/plots/',
help = 'path of folder where plots will be saved')
parser.add_argument('--extension', type = str, default = 'png', choices = ['pdf', 'png', 'jpg'],
help = 'extension of plots')
parser.add_argument('--kl', action = "store_true",
help = 'calculate gaussian-based kl divergences?')
parser.add_argument('--tmp', action = "store_true",
help = 'use tmp files?')
parser.add_argument('-v', '--verbose', action = "store_true",
help = 'if specified, messages will br printed throughout plotting')

args = parser.parse_args()


####################################
####################################
# auxiliary functions         ######
####################################
####################################

# load the network
df = pd.read_csv('fb2.txt', delim_whitespace=True, )

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
KK = max(v1.max(), v2.max())+1
X = np.zeros((KK,KK))
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

def predict(alphs, gams, lambs, Ths, N, T, verbose = True):
    K = Ths.shape[1]
    nEVs = []
    HEs = []
    for t in range(T):
        #if t%5==0: print('obs ' + str(t))
        if verbose: print(str(t+1) + '/' + str(T))
        m = np.random.randint(0, Ths.shape[0], 1)

        ######################
        # simulate from hypers
        #alph = alphs[m]
        #gam = gams[m]
        #lamb = lambs[m]
        #Ths[m] = rej_beta(Ths.shape[1], alph, gam, lamb)
        # to simulate from Ths instead, comment the above block out
        ######################

        # generate N rounds of bernoulli at each pair of vertices using a binomial
        W = np.outer(Ths[m], Ths[m])
        W[np.arange(K), np.arange(K)] = 0
        X = np.random.binomial(N, W, size=W.shape)

        # collect nonzero edge vertex pairs
        V1, V2 = np.where(X>0)

        # create binary sequence of edges for each nonzero pair
        nzEs = np.zeros((V1.shape[0], N))
        for i in range(V1.shape[0]):
            N_pair = X[V1[i], V2[i]]
            idcs = np.random.choice(np.arange(N), size=N_pair, replace=False)
            nzEs[i, idcs] = 1
        # compute number of edges in each round, then cumsum
        nE = np.cumsum(nzEs.sum(axis=0))
        # compute number of unique vertices using a set
        nV = np.zeros(N)
        V = set()
        for n in range(N):
            idcs = np.where(nzEs[:, n]>0)[0]
            for i in idcs:
                V.add(V1[i])
                V.add(V2[i])
            nV[n] = len(V)
        nEVs.append(np.vstack((np.log10(nE), np.log10(nV))))

        # compute the degrees of the final network
        degrees = (X+X.T).sum(axis=0)
        #degrees = np.sort(degrees)
        hist, edges = np.histogram(degrees, density=True, bins=50)
        #hist = hist/hist.sum()
        edges = edges[1:]
        HEs.append([edges, np.log10(hist)])

    return nEVs, HEs


# FOLDER SETTINGS
path = args.outpath
# check if necessary folder structure exists, and create it if it doesn't
if not os.path.exists(path): os.makedirs(path)


# other arg parse variables
inpath = args.inpath
extension = args.extension
calculate_kl = args.kl
tmp = args.tmp
verbose = args.verbose

if verbose:
    print('network model plotting')
    print()


# import density, sampler, and rbf kernel
from kernels.network import kernel_sampler
from targets.networkpdf import logp_aux
from RKHSkernels.rbf import *
kernel, dk_x, dk_y, dk_xy = get_kernel(700)


# define densities and up
def logp(x): return logp_aux(x, 1)
def p(x): return np.exp(logp(x))
sp = egrad(logp)
up = lbvi.up_gen(kernel, sp, dk_x, dk_y, dk_xy)


# import lbvi data
load_path = inpath
if tmp:
    load_path += 'tmp_'
    extra = ''
else:
    extra = '_1_0.0'
y = np.load('initial_sample.npy')
T = np.load(load_path + 'T' + extra + '.npy')
w = np.load(load_path + 'w' + extra + '.npy')
obj = np.load(load_path + 'obj' + extra + '.npy')
cput = np.load(load_path + 'cput' + extra + '.npy')
kernels = np.load(load_path + 'kernels' + extra + '.npy')

# import mcmc data
mcmc_sample = np.load('mcmc_sample.npy')

# print details
if verbose:
    print('initial alphas: ' + str(y[:,0]))
    print('initial gammas: ' + str(y[:,1]))
    print('initial lambdas: ' + str(y[:,2]))
    print()
    print('weights: ' + str(w))
    print('number of steps: ' + str(T))
    print('ksd trace: ' + str(obj))
    print()


# define colors and settings
lbvi_color = '#39558CFF'
lbvi_color2 = '#56C667FF'
mcmc_color = '#F79044FF'
normal_linewidth = 4
plt_alpha = 0.5
legend_fontsize = 'medium'

# generate lbvi sample or load it if it has been generated before (to speed plot tweaking)
if os.path.exists(path + 'lbvi_sample.npy'):
    lbvi_sample = np.load(path + 'lbvi_sample.npy')
    if verbose:
        print('loading existing lbvi sample')
        print()
        print('begin plotting!')
else:
    if verbose: print('generating lbvi sample; this might take a while')

    np.random.seed(123)
    size = 1000
    sizes = np.floor(size*w).astype(int)
    sizes[0] += size-sizes.sum()
    K = 2010
    lbvi_sample = np.zeros((1,K+3))
    alpha_init = y[:,0]
    gam_init = y[:,1]
    lamb_init = y[:,2]
    Th_init = y[:,3:]


    # sample
    for i in range(alpha_init.shape[0]):
        if verbose: print(str(i+1) + '/7')
        if w[i] == 0: continue
        tmp = np.zeros((sizes[i],K+3))

        if verbose: print('sampling from posterior')
        for s in range(sizes[i]):
            if verbose: print(str(s+1) + '/' + str(sizes[i]), end = '\r')
            Alphs, Gams, Lambs, Ths = adaptive_truncation_sampler(T[i], Obs, NN, K,
                                            alph=alpha_init[i], lamb=lamb_init[i], gam=gam_init[i], Th0 = Th_init[i,:],
                                            alpha_step = 0.04, lambda_step = 0.1,
                                            Th0_step = 0.03, Th1_step = 0.1, ThK_step = 0.1, verbose = False)
            # save in array
            tmp[s,0] = Alphs[-1]
            tmp[s,1] = Gams[-1]
            tmp[s,2] = Lambs[-1]
            tmp[s,3:] = Ths[-1,:]
        # end for

        lbvi_sample = np.concatenate((lbvi_sample, tmp))
    # end for

    lbvi_sample = lbvi_sample[1:,:] # rm row with zeros

    #lbvi_sample = lbvi.mix_sample(1000, y = y, T = T+1, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = 1)
    np.save(path + 'lbvi_sample.npy', lbvi_sample)
    if verbose:
        print('lbvi sample generated')
        print()
        print('begin plotting!')

print('lbvi sample: ' + str(lbvi_sample))
####################################
####################################
# alpha vs gamma scatter plot ######
####################################
####################################

plt.scatter(lbvi_sample[:,0],lbvi_sample[:,2], label = 'LBVI', c = lbvi_color, alpha = plt_alpha)
plt.scatter(mcmc_sample[:,0],mcmc_sample[:,2], label = 'MCMC', c = mcmc_color, alpha = plt_alpha)

# add axes labels etc
plt.xlabel('α')
#plt.xlim(0,1)
plt.ylabel('λ')
plt.legend(fontsize = legend_fontsize)
plt.savefig(path + 'network_alpha_lambda_scatter.' + extension, dpi=900, bbox_inches='tight')
plt.clf()


####################################
####################################
# lambda histogram            ######
####################################
####################################

plt.hist(lbvi_sample[:,1], bins = 30, density = True, alpha = plt_alpha, facecolor = lbvi_color, edgecolor='black')
plt.hist(mcmc_sample[:,1], bins = 30, density = True, alpha = plt_alpha, facecolor = mcmc_color, edgecolor='black')

# add axes labels etc
plt.xlabel('γ')
plt.xlim(1.8,3.2)
plt.savefig(path + 'network_gamma_hist.' + extension, dpi=900, bbox_inches='tight')
plt.clf()



####################################
####################################
# posterior predictive plots  ######
####################################
####################################
if verbose: print('begin posterior predictive plots')

# sample from posterior predictive
T = 15
pp_alpha = 0.3
if verbose: print('sampling ' + str(T) + ' obs from the posterior predictive from MCMC')
mcmc_lEVs, mcmc_HEs = predict(mcmc_sample[:,0], mcmc_sample[:,1], mcmc_sample[:,2], mcmc_sample[:,3:], NN, T)
lbvi_dense = lbvi_sample[lbvi_sample[:,0]<0.1,:]
if verbose: print('sampling ' + str(T) + ' obs from the posterior predictive from dense LBVI')
lbvi_dense_lEVs, lbvi_dense_HEs = predict(lbvi_dense[:,0], lbvi_dense[:,1], lbvi_dense[:,2], lbvi_dense[:,3:], NN, T)
lbvi_sparse = lbvi_sample[lbvi_sample[:,0]>=0.1,:]
if verbose: print('sampling ' + str(T) + ' obs from the posterior predictive from sparse LBVI')
lbvi_sparse_lEVs, lbvi_sparse_HEs = predict(lbvi_sparse[:,0], lbvi_sparse[:,1], lbvi_sparse[:,2], lbvi_sparse[:,3:], NN, T)

# log edges ##################################
plt.clf()
plt.plot(np.log10(nE), np.log10(nV), lw=5, color='black', label='Observation')

# plot first for labels
t = 0
plt.plot(lbvi_dense_lEVs[t][0], lbvi_dense_lEVs[t][1], lw=2, color=lbvi_color, alpha = pp_alpha, label='LBVI dense')
plt.plot(lbvi_sparse_lEVs[t][0], lbvi_sparse_lEVs[t][1], lw=2, color=lbvi_color2, alpha = pp_alpha, label='LBVI sparse')
plt.plot(mcmc_lEVs[t][0], mcmc_lEVs[t][1], lw=2, color=mcmc_color, alpha = pp_alpha, label='MCMC')

# plot rest without labels
for t in range(1,T):
    plt.plot(lbvi_dense_lEVs[t][0], lbvi_dense_lEVs[t][1], lw=2, color=lbvi_color, alpha = pp_alpha)
    plt.plot(lbvi_sparse_lEVs[t][0], lbvi_sparse_lEVs[t][1], lw=2, color=lbvi_color2, alpha = pp_alpha)
    plt.plot(mcmc_lEVs[t][0], mcmc_lEVs[t][1], lw=2, color=mcmc_color, alpha = pp_alpha)

# add axes labels etc
plt.xlabel('Log10(E)')
plt.ylabel('Log10(V)')
plt.legend(fontsize = legend_fontsize)
plt.savefig(path + 'loge_logv.' + extension, dpi=900, bbox_inches='tight')
plt.clf()

# log vertices ##################################
plt.plot(edges, np.log10(hist), lw=5, color='black', label='Observation')

# plot first for labels
t = 0
plt.plot(lbvi_dense_HEs[t][0], lbvi_dense_HEs[t][1], lw=2, color=lbvi_color, alpha = pp_alpha, label='LBVI dense')
plt.plot(lbvi_sparse_HEs[t][0], lbvi_sparse_HEs[t][1], lw=2, color=lbvi_color2, alpha = pp_alpha, label='LBVI sparse')
plt.plot(mcmc_HEs[t][0], mcmc_HEs[t][1], lw=2, color=mcmc_color, alpha = pp_alpha, label='MCMC')

# plot rest without labels
for t in range(T):
    plt.plot(lbvi_dense_HEs[t][0], lbvi_dense_HEs[t][1], lw=2, color=lbvi_color, alpha = pp_alpha)
    plt.plot(lbvi_sparse_HEs[t][0], lbvi_sparse_HEs[t][1], lw=2, color=lbvi_color2, alpha = pp_alpha)
    plt.plot(mcmc_HEs[t][0], mcmc_HEs[t][1], lw=2, color=mcmc_color, alpha = pp_alpha)

# add axes labels etc
plt.xlabel('Degree')
plt.xlim(0,500)
plt.ylabel('log Density')
plt.legend(fontsize = legend_fontsize)
plt.savefig(path + 'degree_logd.' + extension, dpi=900, bbox_inches='tight')
plt.clf()




if verbose:
    print('done plotting!')
    print()





####################################
####################################
# KL divergence calculation   ######
####################################
####################################

if calculate_kl:
    if verbose:
        print('calculating kl divergences')
        print()

    # mcmc gaussian approximation
    mcmc_mean = np.mean(mcmc_sample, axis=0)
    mcmc_cov = np.var(mcmc_sample, axis=0)
    mcmc_gaussian_logpdf = lambda x : -0.5*x.shape[1]*np.log(2*np.pi) - 0.5*np.sum(np.log(mcmc_cov)) - 0.5*np.sum((x-mcmc_mean)**2 / mcmc_cov, axis=-1)
    mcmc_lp = mcmc_gaussian_logpdf(lbvi_sample) # to only compute it once

    # lbvi gaussian mixture
    # generate lbvi sample or load it if it has been generated before (to speed plot tweaking)
    # this sample is just to define the means and variances of the lbvi gaussian mixture
    if os.path.exists(path + 'lbvi_tmp_sample.npy'):
        tmp_sample = np.load(path + 'lbvi_tmp_sample.npy')
        if verbose: print('loading existing lbvi sample')
    else:
        if verbose: print('generating lbvi sample; this might take a while')
        tmp_sample = kernel_sampler(y, np.maximum(1,T), S = 100, logp = logp, t_increment = 1)
        np.save(path + 'lbvi_tmp_sample.npy', tmp_sample)
        if verbose: print('lbvi sample generated')

    # extract details from sample
    lbvi_mus = np.mean(tmp_sample, axis=0)
    lbvi_cov = np.var(tmp_sample, axis=0)

    # define logpdf with logsumexp trick
    def lbvi_gaussian_logpdf(x):
        N = lbvi_mus.shape[0]
        exponents = np.zeros((N,x.shape[0]))
        for n in range(N):
            if w[n] == 0: continue # components with weight 0 do not contribute to the mixture
            tmp_mean = lbvi_mus[n,:]
            tmp_cov = lbvi_cov[n,:]
            exponents[n,:] = np.log(w[n]) - 0.5*x.shape[1]*np.log(2*np.pi) - 0.5*np.sum(np.log(tmp_cov)) - 0.5*np.sum((x-tmp_mean)**2 / tmp_cov, axis=-1)
        # end for
        lmax = np.amax(exponents, axis = 0)
        return lmax + np.log(np.sum(np.exp(exponents - lmax), axis = 0))
    #lbvi_gaussian_sample = lambda size : bvi.mixture_sample(size, mus = lbvi_mus, sqrtSigmas = np.sqrt(lbvi_cov), alphas = w)
    lbvi_lp = lbvi_gaussian_logpdf(lbvi_sample) # to only compute it once

    # kl estimation via importance sampling
    lp = logp(lbvi_sample)
    #print('lp: ' + str(lp))
    #print('mcmc lp: ' + str(mcmc_lp))
    #print('lbvi lp: ' + str(lbvi_lp))
    #print()
    #print('p: ' + str(np.exp((lp))))
    #print('mcmc p: ' + str(np.exp(mcmc_lp)))
    #print('lbvi p: ' + str(np.exp(lbvi_lp)))
    #print()

    kl_rev_mcmc = np.mean((np.exp(mcmc_lp) / np.exp(lbvi_lp)) * (mcmc_lp - lp))
    kl_rev_lbvi = np.mean(lbvi_lp - lp)
    kl_fwd_mcmc = np.mean((np.exp(lp) / np.exp(lbvi_lp)) * (lp - mcmc_lp))
    kl_fwd_lbvi = np.mean((np.exp(lp) / np.exp(lbvi_lp)) * (lp - lbvi_lp))

    if verbose:
        print('KL(q_mcmc || p) = ' + str(kl_rev_mcmc))
        print('KL(q_lbvi || p) = ' + str(kl_rev_lbvi))
        print('KL(p || q_mcmc) = ' + str(kl_fwd_mcmc))
        print('KL(p || q_lbvi) = ' + str(kl_fwd_lbvi))
