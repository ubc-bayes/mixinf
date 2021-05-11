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
plt.rcParams.update({'font.size': 16})
import argparse
import sys, os
import imageio
import pickle as pk

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../lbvi/'))
import lbvi # functions to do locally-adapted boosting variational inference
import bvi
import ubvi

# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="plot comparison between lbvi and other vi and mcmc methodologies")

parser.add_argument('--inpath', type = str, default = 'results/',
help = 'path of folder where csv files are stored')
parser.add_argument('--outpath', type = str, default = 'results/plots/',
help = 'path of folder where plots will be saved')
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy', '5-mixture'],
help = 'target distribution to use')
parser.add_argument('--reps', type = int, default = 1,
help = 'number of times each method was run')
parser.add_argument('--tol', type = float, nargs = '+',
help = 'sequence of step size tolerances')
parser.add_argument('--lbvi', action = "store_true",
help = 'plot lbvi?')
parser.add_argument('--kernel', type = str, default = 'gaussian', choices=['gaussian'],
help = 'kernel to use in mixtures')
parser.add_argument('--rkhs', type = str, default = 'rbf', choices=['rbf'],
help = 'RKHS kernel to use')
parser.add_argument('--ubvi', action = "store_true",
help = 'plot ubvi?')
parser.add_argument('--bvi', action = "store_true",
help = 'plot bbbvi?')
parser.add_argument('--gvi', action = "store_true",
help = 'plot standard gaussian vi?')
parser.add_argument('--hmc', action = "store_true",
help = 'plot hamiltonian monte carlo?')
parser.add_argument('--rwmh', action = "store_true",
help = 'plot random-walk metropolis-hastings?')
parser.add_argument('--extension', type = str, default = 'png', choices = ['pdf', 'png', 'jpg'],
help = 'extension of plots')

args = parser.parse_args()


# FOLDER SETTINGS
path = args.outpath

# check if necessary folder structure exists, and create it if it doesn't
if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(path + 'densities/')
    os.makedirs(path + 'logdensities/')

# other arg parse variables
inpath = args.inpath
extension = args.extension
reps = args.reps
tols = np.array(args.tol)

# import flags
lbvi_flag = args.lbvi
ubvi_flag = args.ubvi
bvi_flag = args.bvi
gvi_flag = args.gvi
hmc_flag = args.hmc
rwmh_flag = args.rwmh


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
if target == 'banana-gaussian':
    from targets.banana_gaussian import *
    xlim = np.array([-3, 3])
    ylim = np.array([-2, 3])

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

# get number of repetitions in simulation
reps = len(glob.glob(inpath + 'settings*'))

# init number of kernels for plotting later
lbvi_kernels = np.array([])
ubvi_kernels = np.array([])
bvi_kernels = np.array([])

# PLOT ####
print('begin plotting!')

if ubvi_flag:
    # load ubvi results
    f = open(inpath + 'ubvi/results.pk', 'rb')
    ubvis = pk.load(f)
    f.close()

for r in range(reps):

    if lbvi_flag:
        # retrieve lbvi settings
        tmp_path = inpath + 'lbvi/'
        y = np.load(tmp_path + 'y_' + str(r+1) + '.npy')
        w = np.load(tmp_path + 'w_' + str(r+1) + '.npy')
        T = np.load(tmp_path + 'T_' + str(r+1) + '.npy')
        lbvi_kernels = np.append(lbvi_kernels, w[w > 0].shape[0])


    if ubvi_flag:
        # retrieve ubvi results
        ubvi_mu = ubvis[0][r]['mus']
        ubvi_Sig = ubvis[0][r]['Sigs']
        ubvi_wt = ubvis[0][r]['weights']
        ubvi_kernels = np.append(ubvi_kernels, ubvi_wt[ubvi_wt > 0].shape[0])


    if bvi_flag:
        # retrieve bvi settings and build sqrt matrices
        tmp_path = inpath + 'bvi/'
        mus = np.load(tmp_path + 'means_' + str(r+1) + '.npy')
        Sigmas = np.load(tmp_path + 'covariances_' + str(r+1) + '.npy')
        alphas = np.load(tmp_path + 'weights_' + str(r+1) + '.npy')
        bvi_kernels = np.append(bvi_kernels, alphas[alphas > 0].shape[0])


    if gvi_flag:
        # retrieve gvi settings
        tmp_path = inpath + 'gvi/'
        mu = np.load(tmp_path + 'mean_' + str(r+1) + '.npy')
        Sigma = np.load(tmp_path + 'covariance_' + str(r+1) + '.npy')
        SigmaInv = np.load(tmp_path + 'inv_covariance_' + str(r+1) + '.npy')
        SigmaLogDet = np.load(tmp_path + 'logdet_covariance_' + str(r+1) + '.npy')

    if hmc_flag:
        # retrieve hmc sample
        tmp_path = inpath + 'hmc/'
        hmc = np.load(tmp_path + 'y_' + str(r+1) + '.npy')


    if rwmh_flag:
        # retrieve rwmh sample
        tmp_path = inpath + 'rwmh/'
        rwmh = np.load(tmp_path + 'y_' + str(r+1) + '.npy')

    # DENSITY PLOT
    # initialize plot with target density contour
    xx = np.linspace(xlim[0], xlim[1], 1000)
    yy = np.linspace(ylim[0], ylim[1], 1000)
    tt = np.array(np.meshgrid(xx, yy)).T.reshape(1000**2, 2)
    f = np.exp(logp(tt)).reshape(1000, 1000).T
    fig,ax=plt.subplots(1,1)
    cp = ax.contour(xx, yy, f, label = 'Target')
    #fig.colorbar(cp)

    if rwmh_flag:
        # add rwmh samples
        plt.scatter(rwmh[:,0], rwmh[:,1], marker='.', c='khaki', alpha = 0.9, label = 'RWMH')

    if hmc_flag:
        # add rwmh histogram
        plt.scatter(hmc[:,0], hmc[:,1], marker='.', c='deeppink', alpha = 0.9, label = 'HMC')

    if lbvi_flag:
        # add lbvi samples
        kk = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
        plt.scatter(kk[:,0], kk[:,1], marker='.', c='b', alpha = 0.4, label = 'LBVI')

    if ubvi_flag:
        # add ubvi density
        xxx, yyy = np.meshgrid(xx, yy)
        X = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
        lq_ubvi = ubvi.mixture_logpdf(X, ubvi_mu, ubvi_Sig, ubvi_wt)
        lq_ubvi = lq_ubvi.reshape(1000,1000)
        cp = ax.contour(xxx, yyy, lq_ubvi, label = 'UBVI', cmap = 'cividis')

    if bvi_flag:
        # add bvi density
        bvi_logq = lambda x : bvi.mixture_logpdf(x, mus, Sigmas, alphas)
        f = np.exp(bvi_logq(tt)).reshape(1000, 1000).T
        cp = ax.contour(xx, yy, f, label = 'BBBVI', cmap = 'inferno')

    if gvi_flag:
        # add gvi density
        gvi_logq = lambda x : -0.5*2*np.log(2*np.pi) - 0.5*2*np.log(SigmaLogDet) - 0.5*((x-mu).dot(SigmaInv)*(x-mu)).sum(axis=-1)
        f = np.exp(gvi_logq(tt)).reshape(1000, 1000).T
        cp = ax.contour(xx, yy, f, label = 'BBBVI', cmap = 'magma')


    # add labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    #plt.title('Density comparison')

    # save plot
    plt.savefig(path + 'density_comparison'  + str(r+1) + '.' + extension, dpi=900, bbox_inches='tight')
    plt.clf()
    ##########################


# TIMES PLOT #################
if lbvi_flag and bvi_flag and ubvi_flag:

    # retrieve lbvi times
    lbvi_times_dir = glob.glob(inpath + 'times/lbvi*')
    lbvi_times = np.array([])
    for file in lbvi_times_dir:
        lbvi_times = np.append(lbvi_times, np.load(file))

    # retrieve ubvi times
    ubvi_times_dir = glob.glob(inpath + 'times/ubvi*')
    ubvi_times = np.array([])
    for file in ubvi_times_dir:
        ubvi_times = np.append(ubvi_times, np.load(file))

    # retrieve bvi times
    bvi_times_dir = glob.glob(inpath + 'times/bvi*')
    bvi_times = np.array([])
    for file in bvi_times_dir:
        bvi_times = np.append(bvi_times, np.load(file))

    # merge in data frame
    times = pd.DataFrame({'method' : np.append(np.repeat('LBVI', lbvi_times.shape[0]), np.append(np.repeat('UBVI', ubvi_times.shape[0]), np.repeat('BVI', bvi_times.shape[0]))), 'time' : np.append(lbvi_times, np.append(ubvi_times, bvi_times))})


    # plot
    plt.clf()
    fig, ax1 = plt.subplots()
    times.boxplot(column = 'time', by = 'method', grid = False)
    plt.xlabel('Method')
    plt.ylabel('Running time (s)')
    plt.title('')
    plt.suptitle('')
    plt.savefig(path + 'times.' + extension, dpi=900, bbox_inches='tight')
###################

# NON-ZERO KERNELS PLOT
if lbvi_flag and bvi_flag and ubvi_flag:

    # merge in data frame
    kernels = pd.DataFrame({'method' : np.append(np.repeat('LBVI', lbvi_kernels.shape[0]), np.append(np.repeat('UBVI', bvi_kernels.shape[0]), np.repeat('BVI', bvi_kernels.shape[0]))), 'kernels' : np.append(lbvi_kernels, np.append(ubvi_kernels, bvi_kernels))})

    # plot
    plt.clf()
    fig, ax1 = plt.subplots()
    kernels.boxplot(column = 'kernels', by = 'method', grid = False)
    plt.xlabel('Method')
    plt.ylabel('Number of non-zero kernels')
    plt.title('')
    plt.suptitle('')
    plt.savefig(path + 'kernels.' + extension, dpi=900, bbox_inches='tight')
###################

print('done plotting!')