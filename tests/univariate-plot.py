# plot the results of univariate simulation stored in .cvs files in given folder

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

# other arg parse variables
inpath = args.inpath
extension = args.extension

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
    xlim = np.array([-6, 6]) # for plotting
if target == 'cauchy':
    from targets.cauchy import *
    xlim = np.array([-10, 10])
if target == '5-mixture':
    from targets.fivemixture import *
    xlim = np.array([-3, 15]) # for plotting


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
        hmc = np.squeeze(np.load(tmp_path + 'y_' + str(r+1) + '.npy'), axis=1)


    if rwmh_flag:
        # retrieve rwmh sample
        tmp_path = inpath + 'rwmh/'
        rwmh = np.squeeze(np.load(tmp_path + 'y_' + str(r+1) + '.npy'), axis=1)


    # LOG DENSITY PLOT
    # initialize plot with target log density
    t = np.linspace(xlim[0], xlim[1], 2000)
    f = logp(t[:,np.newaxis])
    plt.plot(t, f, 'k-', label = 'Target', linewidth = 1, markersize = 1.5)

    if lbvi_flag:
        # add lbvi log density based on kde
        kk = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
        yy = stats.gaussian_kde(np.squeeze(kk), bw_method = 0.15).evaluate(t)
        plt.plot(t, np.log(yy), '--b', label = 'LBVI')

    if ubvi_flag:
        # add ubvi log density
        lq = ubvi.mixture_logpdf(t[:, np.newaxis], ubvi_mu, ubvi_Sig, ubvi_wt)
        plt.plot(t, lq, linestyle = 'dashed', color = 'salmon', label='UBVI')

    if bvi_flag:
        # add bvi log density
        bvi_logq = lambda x : bvi.mixture_logpdf(x, mus, Sigmas, alphas)
        plt.plot(t, bvi_logq(t[:,np.newaxis]), '--m', label='BBBVI')

    if gvi_flag:
        # add gvi log density
        gvi_logq = lambda x : -0.5*1*np.log(2*np.pi) - 0.5*1*np.log(SigmaLogDet) - 0.5*((x-mu).dot(SigmaInv)*(x-mu)).sum(axis=-1)
        plt.plot(t, gvi_logq(t[:,np.newaxis]), linestyle = 'dashed', color = 'forestgreen', label='GVI')

    if hmc_flag:
        # add rwmh log density based on kde
        yy = stats.gaussian_kde(hmc, bw_method = 0.15).evaluate(t)
        plt.plot(t, np.log(yy), linestyle = 'dashed', color = 'deeppink', label = 'HMC')


    if rwmh_flag:
        # add rwmh log density based on kde
        yy = stats.gaussian_kde(rwmh, bw_method = 0.15).evaluate(t)
        plt.plot(t, np.log(yy), linestyle = 'dashed', color = 'khaki', label = 'RWMH')

    # add labels
    plt.xlabel('x')
    plt.ylabel('Log-density')
    #plt.title('Log-density comparison')
    plt.xlim(xlim)
    plt.legend()

    # save plot
    plt.savefig(path + 'log-density_comparison' + str(r+1) + '.' + extension, dpi=900, bbox_inches='tight')
    plt.clf()
    ##########################


    # DENSITY PLOT
    # initialize plot with target density
    t = np.linspace(xlim[0], xlim[1], 2000)
    f = p(t[:,np.newaxis])
    plt.plot(t, f, 'k-', label = 'Target', linewidth = 1, markersize = 1.5)

    if rwmh_flag:
        # add rwmh histogram
        plt.hist(rwmh, label = 'RWMH', density = True, bins = 50, alpha = 0.3, facecolor = 'khaki', edgecolor='black')

    if hmc_flag:
        # add rwmh histogram
        plt.hist(hmc, label = 'HMC', density = True, bins = 50, alpha = 0.3, facecolor = 'deeppink', edgecolor='black')

    if lbvi_flag:
        # add lbvi histogram
        kk = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
        plt.hist(kk, label = 'LBVI', density = True, bins = 50, alpha = 0.3, facecolor = 'blue', edgecolor='black')

    if ubvi_flag:
        # add ubvi density
        plt.plot(t, np.exp(lq), linestyle = 'dashed', color = 'salmon', label='UBVI')

    if bvi_flag:
        # add bvi density
        plt.plot(t, np.exp(bvi_logq(t[:,np.newaxis])), '--m', label='BBBVI')

    if gvi_flag:
        # add gvi density
        plt.plot(t, np.exp(gvi_logq(t[:,np.newaxis])), linestyle = 'dashed', color = 'forestgreen', label='GVI')


    # add labels
    plt.xlabel('x')
    plt.ylabel('Density')
    #plt.title('Density comparison')
    plt.xlim(xlim)
    plt.legend()

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
