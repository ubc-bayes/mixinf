# plot the results of bivariate simulation stored in .cvs files in given folder

# PREAMBLE ####
import glob
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import argparse
import sys, os
import imageio

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../lbvi/'))
import lbvi # functions to do locally-adapted boosting variational inference
import bvi

# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="plot comparison between lbvi and bvi")

parser.add_argument('--inpath', type = str, default = 'results/',
help = 'path of folder where csv files are stored')
parser.add_argument('--outpath', type = str, default = 'results/plots/',
help = 'path of folder where plots will be saved')
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy', '5-mixture', 'banana', 'double-banana'],
help = 'target distribution to use')
parser.add_argument('--kernel', type = str, default = 'gaussian', choices=['gaussian'],
help = 'kernel to use in mixtures')
parser.add_argument('--rkhs', type = str, default = 'rbf', choices=['rbf'],
help = 'RKHS kernel to use')
parser.add_argument('--extension', type = str, default = 'png', choices = ['pdf', 'png', 'jpg'],
help = 'extension of plots')

args = parser.parse_args()


# FOLDER SETTINGS
path = args.outpath

# check if necessary folder structure exists, and create it if it doesn't
if not os.path.exists(path):
    os.makedirs(path)

# other arg parse variables
inpath = args.inpath
extension = args.extension



# IMPORT TARGET DENSITY ####
target = args.target
if target == '4-mixture':
    from targets.fourmixture import *
    xlim = np.array([-6, 6])
    ylim = np.array([-6, 6])
if target == 'cauchy':
    from targets.cauchy import *
    xlim = np.array([-10, 10])
    ylim = np.array([-10, 10])
if target == '5-mixture':
    from targets.fivemixture import *
    xlim = np.array([-3, 15])
    ylim = np.array([-3, 15])
if target == 'banana':
    from targets.banana import *
    xlim = np.array([-15, 15])
    ylim = np.array([-15, 15])
if target == 'double-banana':
    from targets.double_banana import *
    xlim = np.array([-2.5, 2.5])
    ylim = np.array([-1, 1])


# import kernel for mixture
kernel = args.kernel
if kernel == 'gaussian':
    from kernels.gaussian import *

# import RKHS kernel
rkhs = args.rkhs
if rkhs == 'rbf':
    from RKHSkernels.rbf import *


# define densities and up
def logp(x): return logp_aux(x, 1)
def p(x): return np.exp(logp(x))
sp = egrad(logp)
up = lbvi.up_gen(kernel, sp, dk_x, dk_y, dk_xy)




# PLOT ####
print('begin plotting!')


# retrieve lbvi settings
tmp_path = inpath + 'lbvi/'
y = np.load(tmp_path + 'y.npy')
w = np.load(tmp_path + 'w.npy')
T = np.load(tmp_path + 'T.npy')




# retrieve bvi settings and build sqrt matrices
tmp_path = inpath + 'bvi/'
mus = np.load(tmp_path + 'means.npy')
Sigmas = np.load(tmp_path + 'covariances.npy')
alphas = np.load(tmp_path + 'weights.npy')

# build sqrt matrices array
sqrtSigmas = np.zeros(Sigmas.shape)
for i in range(Sigmas.shape[0]):
    sqrtSigmas[i,:,:] = sqrtm(Sigmas[i,:,:])
bvi_logq = lambda x : bvi.mixture_logpdf(x, mus, Sigmas, alphas)

# DENSITY PLOT
# initialize plot with target density contour
xx = np.linspace(xlim[0], xlim[1], 1000)
yy = np.linspace(ylim[0], ylim[1], 1000)
tt = np.array(np.meshgrid(xx, yy)).T.reshape(1000**2, 2)
f = np.exp(logp(tt)).reshape(1000, 1000).T
fig,ax=plt.subplots(1,1)
cp = ax.contour(xx, yy, f, label = 'Target')
#fig.colorbar(cp)

# add lbvi samples
kk = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
plt.scatter(kk[:,0], kk[:,1], marker='.', c='k', alpha = 0.2, label = 'LBVI')

# add bvi density
f = np.exp(bvi_logq(tt)).reshape(1000, 1000).T
#fig,ax=plt.subplots(1,1)
#cp = ax.contour(xx, yy, f, label = 'BBBVI', colors = 'magenta')
cp = ax.contour(xx, yy, f, label = 'BBBVI', cmap = 'inferno')
#fig.colorbar(cp)

# add labels
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
plt.title('Density comparison')

# save plot
plt.savefig(path + 'density_comparison.' + extension, dpi=900)
plt.clf()
##########################



print('done plotting!')
