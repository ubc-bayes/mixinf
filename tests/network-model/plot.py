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
tmp = args.tmp
verbose = args.verbose

if verbose:
    print('begin plotting!')
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

# generate lbvi sample
lbvi_sample = lbvi.mix_sample(1000, y = y, T = T+1, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = 1)



####################################
####################################
# alpha vs gamma scatter plot ######
####################################
####################################

plt.scatter(lbvi_sample[:,0],lbvi_sample[:,1], label = 'LBVI', c = lbvi_color, alpha = plt_alpha)
plt.scatter(mcmc_sample[:,0],mcmc_sample[:,1], label = 'MCMC', c = mcmc_color, alpha = plt_alpha)

# add axes labels etc
plt.xlabel('α')
plt.xlim(0,1)
plt.ylabel('γ')
plt.legend(fontsize = legend_fontsize)
plt.savefig(path + 'network_alpha_gamma_scatter.' + extension, dpi=900, bbox_inches='tight')
plt.clf()




####################################
####################################
# lambda histogram            ######
####################################
####################################

plt.hist(lbvi_sample[:,2], bins = 50, density = True, alpha = plt_alpha, facecolor = lbvi_color, edgecolor='black')
plt.hist(mcmc_sample[:,2], bins = 50, density = True, alpha = plt_alpha, facecolor = mcmc_color, edgecolor='black')

# add axes labels etc
plt.xlabel('λ')
plt.savefig(path + 'network_lambda_hist.' + extension, dpi=900, bbox_inches='tight')
plt.clf()


print('done plotting!')
