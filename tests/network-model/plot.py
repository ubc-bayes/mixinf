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
w = np.array([0.9, 0, 0, 0, 0, 0.1, 0])
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
    lbvi_sample = lbvi.mix_sample(1000, y = y, T = T+1, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = 1)
    np.save(path + 'lbvi_sample.npy', lbvi_sample)
    if verbose:
        print('lbvi sample generated')
        print()
        print('begin plotting!')



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

plt.hist(lbvi_sample[:,1], bins = 50, density = True, alpha = plt_alpha, facecolor = lbvi_color, edgecolor='black')
plt.hist(mcmc_sample[:,1], bins = 20, density = True, alpha = plt_alpha, facecolor = mcmc_color, edgecolor='black')

# add axes labels etc
plt.xlabel('γ')
plt.savefig(path + 'network_gamma_hist.' + extension, dpi=900, bbox_inches='tight')
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
