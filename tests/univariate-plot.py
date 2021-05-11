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

# init number of kernels for plotting later
lbvi_kernels = np.array([])
ubvi_kernels = np.array([])
bvi_kernels = np.array([])

# PLOT ####
print('begin plotting!')

# define colors and settings
lbvi_color = '#39558CFF'
ubvi_color = '#D64B40FF'
bvi_color = '#74D055FF'
gvi_color = '0.6'
hmc_color = '0.7'
rwmh_color = '0.8'
muted_alpha = 0.4 # for toning down less important lines
muted_linewidth = 1
legend_fontsize = 'x-small'


for r in np.arange(reps):
    for tol in tols:

        if lbvi_flag:
            # retrieve lbvi settings
            tmp_path = inpath + 'lbvi/'
            y = np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy')
            w = np.load(tmp_path + 'w_' + str(r+1) + '_' + str(tol) + '.npy')
            T = np.load(tmp_path + 'T_' + str(r+1) + '_' + str(tol) + '.npy')
            lbvi_sample = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)

        if ubvi_flag:
            # retrieve ubvi results
            #n = len(ubvis[r])
            #ubvi_mu = ubvis[r][n-1]['mus']
            #ubvi_Sig = ubvis[r][n-1]['Sigs']
            #ubvi_wt = ubvis[r][n-1]['weights']
            #ubvi_ksd = np.array([ ubvis[r][i]['ksd'] for i in range(len(ubvis[r]))])

            tmp_path = inpath + 'ubvi/'
            ubvi_mu = np.load(tmp_path + 'means_' + str(r+1) + '_' + str(tol) + '.npy')
            ubvi_Sig = np.load(tmp_path + 'covariances_' + str(r+1) + '_' + str(tol) + '.npy')
            ubvi_wt = np.load(tmp_path + 'weights_' + str(r+1) + '_' + str(tol) + '.npy')


        if bvi_flag:
            # retrieve bvi settings and build sqrt matrices
            tmp_path = inpath + 'bvi/'
            mus = np.load(tmp_path + 'means_' + str(r+1) + '_' + str(tol) + '.npy')
            Sigmas = np.load(tmp_path + 'covariances_' + str(r+1) + '_' + str(tol) + '.npy')
            alphas = np.load(tmp_path + 'weights_' + str(r+1) + '_' + str(tol) + '.npy')
            bvi_logq = lambda x : bvi.mixture_logpdf(x, mus, Sigmas, alphas)


        if gvi_flag:
            # retrieve gvi settings
            tmp_path = inpath + 'gvi/'
            mu = np.load(tmp_path + 'mean_' + str(r+1) + '_' + str(tol) + '.npy')
            Sigma = np.load(tmp_path + 'covariance_' + str(r+1) + '_' + str(tol) + '.npy')
            SigmaInv = np.load(tmp_path + 'inv_covariance_' + str(r+1) + '_' + str(tol) + '.npy')
            SigmaLogDet = np.load(tmp_path + 'logdet_covariance_' + str(r+1) + '_' + str(tol) + '.npy')
            gvi_logq = lambda x : -0.5*1*np.log(2*np.pi) - 0.5*1*np.log(SigmaLogDet) - 0.5*((x-mu).dot(SigmaInv)*(x-mu)).sum(axis=-1)


        if hmc_flag:
            # retrieve hmc sample
            tmp_path = inpath + 'hmc/'
            hmc = np.squeeze(np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy'), axis=1)


        if rwmh_flag:
            # retrieve rwmh sample
            tmp_path = inpath + 'rwmh/'
            rwmh = np.squeeze(np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy'), axis=1)


        # LOG DENSITY PLOT
        # initialize plot with target log density
        t = np.linspace(-50, 50, 2000)
        f = logp(t[:,np.newaxis])
        plt.plot(t, f, linestyle = 'solid', color = 'black', label = 'log p(x)', linewidth = 1.75)

        if lbvi_flag:
            # add lbvi log density based on kde
            lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample), bw_method = 0.25).evaluate(t)
            plt.plot(t, np.log(lbvi_kde), linestyle = 'dashed', color = lbvi_color, label = 'LBVI')

        if ubvi_flag:
            # add ubvi log density
            lq = ubvi.mixture_logpdf(t[:, np.newaxis], ubvi_mu, ubvi_Sig, ubvi_wt)
            plt.plot(t, lq, linestyle = 'dashed', color = ubvi_color, label='UBVI')

        if bvi_flag:
            # add bvi log density
            plt.plot(t, bvi_logq(t[:,np.newaxis]), linestyle = 'dashed', color = bvi_color, label='BBBVI')

        if gvi_flag:
            # add gvi log density
            plt.plot(t, gvi_logq(t[:,np.newaxis]), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha)

        if hmc_flag:
            # add rwmh log density based on kde
            hmc_kde = stats.gaussian_kde(hmc, bw_method = 1).evaluate(t)
            plt.plot(t, np.log(hmc_kde), linestyle = 'dashed', color = hmc_color, label = 'HMC', alpha = muted_alpha)


        if rwmh_flag:
            # add rwmh log density based on kde
            rwmh_kde = stats.gaussian_kde(rwmh, bw_method = 0.15).evaluate(t)
            plt.plot(t, np.log(rwmh_kde), linestyle = 'dashed', color = rwmh_color, label = 'RWMH', alpha = muted_alpha)

        # add labels
        plt.xlabel('x')
        plt.ylabel('Log-density')
        #plt.title('Log-density comparison')
        plt.xlim(-50,50)
        plt.legend(fontsize = legend_fontsize)

        # save plot
        plt.savefig(path + 'logdensities/log-density_comparison' + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
        plt.clf()
        ##########################


        # DENSITY PLOT
        # initialize plot with target density
        t = np.linspace(xlim[0], xlim[1], 2000)
        f = p(t[:,np.newaxis])
        plt.plot(t, f, linestyle = 'solid', color = 'black', label = 'p(x)', linewidth = 1.75)

        if lbvi_flag:
            # add lbvi histogram
            #kk = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
            #plt.hist(kk, label = 'LBVI', density = True, bins = 50, alpha = 0.3, facecolor = lbvi_color, edgecolor='black')
            lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample), bw_method = 0.05).evaluate(t)
            plt.plot(t, lbvi_kde, linestyle = 'dashed', color = lbvi_color, label = 'LBVI')

        if ubvi_flag:
            # add ubvi density
            lq = ubvi.mixture_logpdf(t[:, np.newaxis], ubvi_mu, ubvi_Sig, ubvi_wt)
            plt.plot(t, np.exp(lq), linestyle = 'dashed', color = ubvi_color, label='UBVI')

        if bvi_flag:
            # add bvi density
            plt.plot(t, np.exp(bvi_logq(t[:,np.newaxis])), linestyle = 'dashed', color = bvi_color, label='BBBVI')

        if rwmh_flag:
            # add rwmh histogram
            #plt.hist(rwmh, label = 'RWMH', density = True, bins = 50, alpha = 0.3, facecolor = rwmh_color, edgecolor='black')
            rwmh_kde = stats.gaussian_kde(rwmh, bw_method = 0.15).evaluate(t)
            plt.plot(t, rwmh_kde, linestyle = 'dashed', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw = muted_linewidth)

        if hmc_flag:
            # add rwmh histogram
            #plt.hist(hmc, label = 'HMC', density = True, bins = 50, alpha = 0.3, facecolor = hmc_color, edgecolor='black')
            hmc_kde = stats.gaussian_kde(hmc, bw_method = 1).evaluate(t)
            plt.plot(t, hmc_kde, linestyle = 'dashed', color = hmc_color, label = 'HMC', alpha = muted_alpha, lw = muted_linewidth)

        if gvi_flag:
            # add gvi density
            plt.plot(t, np.exp(gvi_logq(t[:,np.newaxis])), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)


        # add labels
        plt.xlabel('x')
        plt.ylabel('Density')
        #plt.title('Density comparison')
        plt.xlim(xlim)
        plt.legend(fontsize = legend_fontsize)

        # save plot
        plt.savefig(path + 'densities/density_comparison'  + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
        plt.clf()
        ##########################

msize = 15
pltalpha = 0.5

# TIMES PLOT #################
if lbvi_flag and bvi_flag and ubvi_flag:
    plt.clf()

    # plot all reps and tols; for the first plot, add labels
    counter = 1
    for r in np.arange(1,reps+1):
        for tol in tols:
            lbvi_times = np.load(inpath + 'lbvi/cput_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            lbvi_obj = np.load(inpath + 'lbvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            if counter == 1:
                plt.scatter(lbvi_times, np.log(lbvi_obj), c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
            else:
                plt.scatter(lbvi_times, np.log(lbvi_obj), c = lbvi_color, s = msize, alpha = pltalpha)

            ubvi_times = np.load(inpath + 'ubvi/cput_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            ubvi_obj = np.load(inpath + 'ubvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            if counter == 1:
                plt.scatter(ubvi_times, np.log(ubvi_obj), c = ubvi_color, label = 'UBVI', s = msize, alpha = pltalpha)
            else:
                plt.scatter(ubvi_times, np.log(ubvi_obj), c = ubvi_color, s = msize, alpha = pltalpha)

            bvi_times = np.load(inpath + 'bvi/cput_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            bvi_obj = np.load(inpath + 'bvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            if counter == 1:
                plt.scatter(bvi_times, np.log(bvi_obj), c = bvi_color, label = 'BVI', s = msize, alpha = pltalpha)
            else:
                plt.scatter(bvi_times, np.log(bvi_obj), c = bvi_color, s = msize, alpha = pltalpha)

            counter += 1


    # add labels and save
    plt.xlabel('CPU time (s)')
    plt.ylabel('log KSD')
    #plt.ylim(0, np.log(0.1))
    plt.title('')
    plt.suptitle('')
    plt.legend(fontsize = legend_fontsize, loc = 'lower right')
    plt.savefig(path + 'times.' + extension, dpi=900, bbox_inches='tight')
###################




# KERNELS PLOT v2 #################
if lbvi_flag and bvi_flag and ubvi_flag:
    plt.clf()

    # plot all reps and tols; for the first plot, add labels
    counter = 1
    for r in np.arange(1,reps+1):
        for tol in tols:
            lbvi_kernels = np.load(inpath + 'lbvi/kernels_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            lbvi_obj = np.load(inpath + 'lbvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            if counter == 1:
                #plt.scatter(np.log(lbvi_obj), lbvi_kernels, c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
                plt.scatter(lbvi_kernels, np.log(lbvi_obj), c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
            else:
                #plt.scatter(np.log(lbvi_obj), lbvi_kernels, c = lbvi_color, s = msize, alpha = pltalpha)
                plt.scatter(lbvi_kernels, np.log(lbvi_obj), c = lbvi_color, s = msize, alpha = pltalpha)

            ubvi_kernels = np.load(inpath + 'ubvi/kernels_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            ubvi_obj = np.load(inpath + 'ubvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            if counter == 1:
                plt.scatter(ubvi_kernels, np.log(ubvi_obj), c = ubvi_color, label = 'UBVI', s = msize, alpha = pltalpha)
            else:
                plt.scatter(ubvi_kernels, np.log(ubvi_obj), c = ubvi_color, s = msize, alpha = pltalpha)

            bvi_kernels = np.load(inpath + 'bvi/kernels_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            bvi_obj = np.load(inpath + 'bvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
            if counter == 1:
                plt.scatter(bvi_kernels, np.log(bvi_obj), c = bvi_color, label = 'BVI', s = msize, alpha = pltalpha)
            else:
                plt.scatter(bvi_kernels, np.log(bvi_obj), c = bvi_color, s = msize, alpha = pltalpha)

            counter += 1


    # add labels and save
    plt.xlabel('Number of non-zero kernels')
    plt.ylabel('log KSD')
    #plt.ylim(0, np.log(0.1))
    plt.title('')
    plt.suptitle('')
    plt.legend(fontsize = legend_fontsize, loc = 'lower right')
    plt.savefig(path + 'kernels.' + extension, dpi=900, bbox_inches='tight')
###################



print('done plotting!')
