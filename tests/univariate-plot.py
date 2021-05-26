# plot the results of univariate simulation stored in .cvs files in given folder

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
import pickle as pk

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../lbvi/'))
import lbvi # functions to do locally-adapted boosting variational inference
import bvi
import ubvi
import numpy as np

# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="plot comparison between lbvi and other vi and mcmc routines")

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
parser.add_argument('--no_dens', action = "store_true",
help = 'skip plots of all densities and log densities?')
parser.add_argument('--lbvi', action = "store_true",
help = 'plot lbvi?')
parser.add_argument('--kernel', type = str, default = 'gaussian', choices=['gaussian'],
help = 'kernel to use in mixtures')
parser.add_argument('--rkhs', type = str, default = 'rbf', choices=['rbf'],
help = 'RKHS kernel to use')
parser.add_argument('--gamma', type = float, default = 1.,
help = 'if rbf kernel is used, the kernel bandwidth')
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
dens_plots = not args.no_dens

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
lbvi_gamma = args.gamma
if rkhs == 'rbf':
    from RKHSkernels.rbf import *
kernel, dk_x, dk_y, dk_xy = get_kernel(lbvi_gamma)

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
gvi_color = '0.2'
hmc_color = '0.3'
rwmh_color = '0.4'
normal_linewidth = 4
muted_alpha = 0.4 # for toning down less important lines
muted_linewidth = 3
#legend_fontsize = 'x-small'
legend_fontsize = 'medium'

if dens_plots:
    for r in np.arange(reps):
        print(str(r+1) + '/'+ str(reps), end='\r')
        for tol in tols:

            if lbvi_flag:
                # retrieve lbvi settings
                tmp_path = inpath + 'lbvi/'
                y = np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy')
                w = np.load(tmp_path + 'w_' + str(r+1) + '_' + str(tol) + '.npy')
                T = np.load(tmp_path + 'T_' + str(r+1) + '_' + str(tol) + '.npy')
                lbvi_sample = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = 1)

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
            plt.plot(t, f, linestyle = 'solid', color = 'black', label = 'log p(x)', lw = normal_linewidth)

            if lbvi_flag:
                # add lbvi log density based on kde
                lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample), bw_method = 0.25).evaluate(t)
                plt.plot(t, np.log(lbvi_kde), linestyle = 'dashed', color = lbvi_color, label = 'LBVI', lw = normal_linewidth)

            if ubvi_flag:
                # add ubvi log density
                lq = ubvi.mixture_logpdf(t[:, np.newaxis], ubvi_mu, ubvi_Sig, ubvi_wt)
                plt.plot(t, lq, linestyle = 'dashed', color = ubvi_color, label='UBVI', lw = normal_linewidth)

            if bvi_flag:
                # add bvi log density
                plt.plot(t, bvi_logq(t[:,np.newaxis]), linestyle = 'dashed', color = bvi_color, label='BBBVI', lw = normal_linewidth)

            if gvi_flag:
                # add gvi log density
                plt.plot(t, gvi_logq(t[:,np.newaxis]), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)

            if hmc_flag:
                # add rwmh log density based on kde
                hmc_kde = stats.gaussian_kde(hmc, bw_method = 0.5).evaluate(t)
                plt.plot(t, np.log(hmc_kde), linestyle = 'dashdot', color = hmc_color, label = 'HMC', alpha = muted_alpha, lw = muted_linewidth)


            if rwmh_flag:
                # add rwmh log density based on kde
                rwmh_kde = stats.gaussian_kde(rwmh, bw_method = 0.5).evaluate(t)
                plt.plot(t, np.log(rwmh_kde), linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw = muted_linewidth)

            # add labels
            #plt.xlabel('x')
            #plt.ylabel('Log-density')
            #plt.title('Log-density comparison')
            plt.xlim(-50,50)
            plt.ylim(-50,5)
            #plt.legend(fontsize = legend_fontsize)

            # save plot
            plt.savefig(path + 'logdensities/log-density_comparison' + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
            plt.clf()
            ##########################


            # DENSITY PLOT
            # initialize plot with target density
            t = np.linspace(xlim[0], xlim[1], 2000)
            f = p(t[:,np.newaxis])
            plt.plot(t, f, linestyle = 'solid', color = 'black', label = 'p(x)', lw = normal_linewidth)

            if lbvi_flag:
                # add lbvi histogram
                #kk = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
                #plt.hist(kk, label = 'LBVI', density = True, bins = 50, alpha = 0.3, facecolor = lbvi_color, edgecolor='black')
                lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample), bw_method = 0.05).evaluate(t)
                plt.plot(t, lbvi_kde, linestyle = 'dashed', color = lbvi_color, label = 'LBVI', lw = normal_linewidth)

            if ubvi_flag:
                # add ubvi density
                lq = ubvi.mixture_logpdf(t[:, np.newaxis], ubvi_mu, ubvi_Sig, ubvi_wt)
                plt.plot(t, np.exp(lq), linestyle = 'dashed', color = ubvi_color, label='UBVI', lw = normal_linewidth)

            if bvi_flag:
                # add bvi density
                plt.plot(t, np.exp(bvi_logq(t[:,np.newaxis])), linestyle = 'dashed', color = bvi_color, label='BBBVI', lw = normal_linewidth)

            if rwmh_flag:
                # add rwmh histogram
                #plt.hist(rwmh, label = 'RWMH', density = True, bins = 50, alpha = 0.3, facecolor = rwmh_color, edgecolor='black')
                rwmh_kde = stats.gaussian_kde(rwmh, bw_method = 0.05).evaluate(t)
                plt.plot(t, rwmh_kde, linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw = muted_linewidth)

            if hmc_flag:
                # add rwmh histogram
                #plt.hist(hmc, label = 'HMC', density = True, bins = 50, alpha = 0.3, facecolor = hmc_color, edgecolor='black')
                hmc_kde = stats.gaussian_kde(hmc, bw_method = 0.05).evaluate(t)
                plt.plot(t, hmc_kde, linestyle = 'dashdot', color = hmc_color, label = 'HMC', alpha = muted_alpha, lw = muted_linewidth)

            if gvi_flag:
                # add gvi density
                plt.plot(t, np.exp(gvi_logq(t[:,np.newaxis])), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)


            # add labels
            #plt.xlabel('x')
            #plt.ylabel('Density')
            #plt.title('Density comparison')
            plt.xlim(xlim)
            plt.legend(fontsize = legend_fontsize, frameon = False)

            # save plot
            plt.savefig(path + 'densities/density_comparison'  + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
            plt.clf()
##########################


msize = 30
pltalpha = 0.75

## TIMES PLOT #################
#if lbvi_flag and bvi_flag and ubvi_flag:
#    plt.clf()
#
#    # plot all reps and tols; for the first plot, add labels
#    counter = 1
#    for r in np.arange(1,reps+1):
#        for tol in tols:
#            lbvi_times = np.load(inpath + 'lbvi/cput_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            lbvi_obj = np.load(inpath + 'lbvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            if counter == 1:
#                plt.scatter(lbvi_times, lbvi_obj, c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
#            else:
#                plt.scatter(lbvi_times, lbvi_obj, c = lbvi_color, s = msize, alpha = pltalpha)
#
#            ubvi_times = np.load(inpath + 'ubvi/cput_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            ubvi_obj = np.load(inpath + 'ubvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            if counter == 1:
#                plt.scatter(ubvi_times, ubvi_obj, c = ubvi_color, label = 'UBVI', s = msize, alpha = pltalpha)
#            else:
#                plt.scatter(ubvi_times, ubvi_obj, c = ubvi_color, s = msize, alpha = pltalpha)
#
#            bvi_times = np.load(inpath + 'bvi/cput_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            bvi_obj = np.load(inpath + 'bvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            if counter == 1:
#                plt.scatter(bvi_times, bvi_obj, c = bvi_color, label = 'BVI', s = msize, alpha = pltalpha)
#            else:
#                plt.scatter(bvi_times, bvi_obj, c = bvi_color, s = msize, alpha = pltalpha)
#
#            counter += 1
#
#
#    # add labels and save
#    plt.xlabel('CPU time (s)')
#    plt.ylabel('KSD')
#    plt.yscale('log')
#    #plt.ylim(0, np.log(0.1))
#    plt.xlim(0,1000)
#    plt.title('')
#    plt.suptitle('')
#    plt.legend(fontsize = legend_fontsize, loc = 'lower right')
#    plt.savefig(path + 'times.' + extension, dpi=900, bbox_inches='tight')
####################
#
#
#
#
## KERNELS PLOT v2 #################
#if lbvi_flag and bvi_flag and ubvi_flag:
#    plt.clf()
#
#    # plot all reps and tols; for the first plot, add labels
#    counter = 1
#    for r in np.arange(1,reps+1):
#        for tol in tols:
#            lbvi_kernels = np.load(inpath + 'lbvi/kernels_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            lbvi_obj = np.load(inpath + 'lbvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            if counter == 1:
#                #plt.scatter(np.log(lbvi_obj), lbvi_kernels, c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
#                plt.scatter(lbvi_kernels, lbvi_obj, c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
#            else:
#                #plt.scatter(np.log(lbvi_obj), lbvi_kernels, c = lbvi_color, s = msize, alpha = pltalpha)
#                plt.scatter(lbvi_kernels, lbvi_obj, c = lbvi_color, s = msize, alpha = pltalpha)
#
#            ubvi_kernels = np.load(inpath + 'ubvi/kernels_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            ubvi_obj = np.load(inpath + 'ubvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            if counter == 1:
#                plt.scatter(ubvi_kernels, ubvi_obj, c = ubvi_color, label = 'UBVI', s = msize, alpha = pltalpha)
#            else:
#                plt.scatter(ubvi_kernels, ubvi_obj, c = ubvi_color, s = msize, alpha = pltalpha)
#
#            bvi_kernels = np.load(inpath + 'bvi/kernels_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            bvi_obj = np.load(inpath + 'bvi/obj_' + str(r) + '_' + str(tol) + '.npy')#[-1]
#            if counter == 1:
#                plt.scatter(bvi_kernels, bvi_obj, c = bvi_color, label = 'BVI', s = msize, alpha = pltalpha)
#            else:
#                plt.scatter(bvi_kernels, bvi_obj, c = bvi_color, s = msize, alpha = pltalpha)
#
#            counter += 1
#
#
#    # add labels and save
#    plt.xlabel('Number of non-zero kernels')
#    plt.ylabel('KSD')
#    plt.yscale('log')
#    #plt.ylim(0, np.log(0.1))
#    plt.xlim(0,20)
#    plt.title('')
#    plt.suptitle('')
#    plt.legend(fontsize = legend_fontsize, loc = 'lower right')
#    plt.savefig(path + 'kernels.' + extension, dpi=900, bbox_inches='tight')
####################
#
#
## TIMES AND KERNELS TOGETHER ###################
##plt.rcParams.update({'font.size': 12})
#if lbvi_flag and bvi_flag and ubvi_flag:
#    plt.clf()
#    fig, (ax1, ax2) = plt.subplots(2)
#
#    # plot all reps and tols; for the first plot, add labels
#    counter = 1
#    for r in np.arange(1,reps+1):
#        for tol in tols:
#            lbvi_times = np.load(inpath + 'lbvi/cput_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            lbvi_kernels = np.load(inpath + 'lbvi/kernels_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            lbvi_obj = np.load(inpath + 'lbvi/obj_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            if counter == 1:
#                ax1.scatter(lbvi_times, lbvi_obj, c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
#            else:
#                ax1.scatter(lbvi_times, lbvi_obj, c = lbvi_color, s = msize, alpha = pltalpha)
#            ax2.scatter(lbvi_kernels, lbvi_obj, c = lbvi_color, s = msize, alpha = pltalpha)
#
#            ubvi_times = np.load(inpath + 'ubvi/cput_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            ubvi_kernels = np.load(inpath + 'ubvi/kernels_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            ubvi_obj = np.load(inpath + 'ubvi/obj_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            if counter == 1:
#                ax1.scatter(ubvi_times, ubvi_obj, c = ubvi_color, label = 'UBVI', s = msize, alpha = pltalpha)
#            else:
#                ax1.scatter(ubvi_times, ubvi_obj, c = ubvi_color, s = msize, alpha = pltalpha)
#            ax2.scatter(ubvi_kernels, ubvi_obj, c = ubvi_color, s = msize, alpha = pltalpha)
#
#            bvi_times = np.load(inpath + 'bvi/cput_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            bvi_kernels = np.load(inpath + 'bvi/kernels_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            bvi_obj = np.load(inpath + 'bvi/obj_' + str(r) + '_' + str(tol) + '.npy')[-1]
#            if counter == 1:
#                ax1.scatter(bvi_times, bvi_obj, c = bvi_color, label = 'BVI', s = msize, alpha = pltalpha)
#            else:
#                ax1.scatter(bvi_times, bvi_obj, c = bvi_color, s = msize, alpha = pltalpha)
#            ax2.scatter(bvi_kernels, bvi_obj, c = bvi_color, s = msize, alpha = pltalpha)
#
#            counter += 1
#
#
#    # add labels and save
#    ax1.set_xlabel('CPU time (s)')
#    ax1.set_ylabel('KSD')
#    ax1.set_yscale('log')
#    #ax1.legend(fontsize = legend_fontsize, loc = 'lower right')
#    ax2.set_xlabel('Number of non-zero kernels')
#    ax2.set_ylabel('KSD')
#    ax2.set_yscale('log')
#    plt.tight_layout()
#    plt.savefig(path + 'times_kernels.' + extension, dpi=900, bbox_inches='tight')
####################
#
#
#
## TIMES AND KERNELS TOGETHER ORIGINAL VERSION ###################
#if lbvi_flag and bvi_flag and ubvi_flag:
#
#    # init arrays
#    lbvi_times = np.zeros((reps, tols.shape[0]))
#    lbvi_kernels = np.zeros((reps,tols.shape[0]))
#
#    ubvi_times = np.zeros((reps,tols.shape[0]))
#    ubvi_kernels = np.zeros((reps,tols.shape[0]))
#
#    bvi_times = np.zeros((reps,tols.shape[0]))
#    bvi_kernels = np.zeros((reps,tols.shape[0]))
#
#    # populate arrays
#    for r in range(reps):
#        for t in range(tols.shape[0]):
#            lbvi_times[r,t] = np.load(inpath + 'lbvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')[-1]
#            lbvi_kernels[r,t] = np.load(inpath + 'lbvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')[-1]
#
#            ubvi_times[r,t] = np.load(inpath + 'ubvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')[-1]
#            ubvi_kernels[r,t] = np.load(inpath + 'ubvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')[-1]
#
#            bvi_times[r,t] = np.load(inpath + 'bvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')[-1]
#            bvi_kernels[r,t] = np.load(inpath + 'bvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')[-1]
#        # end for
#    # end for
#    # create error bars for both plots
#    lbvi_times_median = np.median(lbvi_times, axis=0)
#    lbvi_times_err = np.vstack((lbvi_times_median - np.quantile(lbvi_times, 0.25, axis=0), np.quantile(lbvi_times, 0.75, axis=0) - lbvi_times_median))
#    lbvi_kernels_median = np.median(lbvi_kernels, axis=0)
#    lbvi_kernels_err = np.vstack((lbvi_kernels_median - np.quantile(lbvi_kernels, 0.25, axis=0), np.quantile(lbvi_kernels, 0.75, axis=0) - lbvi_kernels_median))
#
#    ubvi_times_median = np.median(ubvi_times, axis=0)
#    ubvi_times_err = np.vstack((ubvi_times_median - np.quantile(ubvi_times, 0.25, axis=0), np.quantile(ubvi_times, 0.75, axis=0) - ubvi_times_median))
#    ubvi_kernels_median = np.median(ubvi_kernels, axis=0)
#    ubvi_kernels_err = np.vstack((ubvi_kernels_median - np.quantile(ubvi_kernels, 0.25, axis=0), np.quantile(ubvi_kernels, 0.75, axis=0) - ubvi_kernels_median))
#
#    bvi_times_median = np.median(bvi_times, axis=0)
#    bvi_times_err = np.vstack((bvi_times_median - np.quantile(bvi_times, 0.25, axis=0), np.quantile(bvi_times, 0.75, axis=0) - bvi_times_median))
#    bvi_kernels_median = np.median(bvi_kernels, axis=0)
#    bvi_kernels_err = np.vstack((bvi_kernels_median - np.quantile(bvi_kernels, 0.25, axis=0), np.quantile(bvi_kernels, 0.75, axis=0) - bvi_kernels_median))
#
#    # plot
#    plt.clf()
#    fig, (ax1, ax2) = plt.subplots(2)
#    # times error bars
#    ax1.errorbar(tols, lbvi_times_median, yerr = lbvi_times_err, linestyle = 'solid', color = lbvi_color, label='LBVI')
#    ax1.errorbar(tols, ubvi_times_median, yerr = ubvi_times_err, linestyle = 'solid', color = ubvi_color, label='UBVI')
#    ax1.errorbar(tols, bvi_times_median, yerr = bvi_times_err, linestyle = 'solid', color = bvi_color, label='BVI')
#
#    # kernels error bars
#    ax2.errorbar(tols, lbvi_kernels_median, yerr = lbvi_kernels_err, linestyle = 'solid', color = lbvi_color, label='LBVI')
#    ax2.errorbar(tols, ubvi_kernels_median, yerr = ubvi_kernels_err, linestyle = 'solid', color = ubvi_color, label='UBVI')
#    ax2.errorbar(tols, bvi_kernels_median, yerr = bvi_kernels_err, linestyle = 'solid', color = bvi_color, label='BVI')
#
#
#    # add labels and save
#    ax1.set_xlabel('KSD')
#    ax1.set_xscale('log')
#    ax1.set_ylabel('CPU time (s)')
#    #ax1.legend(fontsize = legend_fontsize, loc = 'lower right')
#    ax2.set_xlabel('KSD')
#    ax2.set_xscale('log')
#    ax2.set_ylabel('# of kernels')
#    plt.tight_layout()
#    plt.savefig(path + 'times_kernels_new.' + extension, dpi=900, bbox_inches='tight')
####################


# TIMES AND KERNELS TOGETHER FINAL (?) VERSION ###################
if lbvi_flag and bvi_flag and ubvi_flag:

    # init arrays
    ttols = tols.shape[0]
    niter = 50
    lbvi_times = np.zeros((reps*ttols, niter))
    lbvi_kernels = np.zeros((reps*ttols, niter))
    lbvi_ksd = np.zeros((reps*ttols, niter))

    ubvi_times = np.zeros((reps*ttols, niter))
    ubvi_kernels = np.zeros((reps*ttols, niter))
    ubvi_ksd = np.zeros((reps*ttols, niter))

    bvi_times = np.zeros((reps*ttols, niter))
    bvi_kernels = np.zeros((reps*ttols, niter))
    bvi_ksd = np.zeros((reps*ttols, niter))

    # populate arrays
    counter = 0
    for r in range(reps):
        for t in range(tols.shape[0]):
            cput = np.load(inpath + 'lbvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            tmp_n = cput.shape[0]#+1 # sometimes there will be fewer than niter iterations, so get how many there were and substitute
            lbvi_times[counter,:tmp_n] = cput
            lbvi_kernels[counter,:tmp_n] = np.load(inpath + 'lbvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            lbvi_ksd[counter,:tmp_n] = np.load(inpath + 'lbvi/obj_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]

            cput = np.load(inpath + 'ubvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_n = cput.shape[0]
            ubvi_times[counter,:tmp_n] = cput
            ubvi_kernels[counter,:tmp_n] = np.load(inpath + 'ubvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            ubvi_ksd[counter,:tmp_n] = np.load(inpath + 'ubvi/obj_' + str(r+1) + '_' + str(tols[t]) + '.npy')

            cput = np.load(inpath + 'bvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_n = cput.shape[0]
            bvi_times[counter,:tmp_n] = cput
            bvi_kernels[counter,:tmp_n] = np.load(inpath + 'bvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            bvi_ksd[counter,:tmp_n] = np.load(inpath + 'bvi/obj_' + str(r+1) + '_' + str(tols[t]) + '.npy')

            counter += 1
        # end for
    # end for


    # create error bars for all plots via masked arrays (to get rid of placeholder 0's where there were no iterations)
    # LBVI
    lbvi_times_masked = npc.ma.masked_where(lbvi_times == 0, lbvi_times) # mask 0's
    lbvi_times_median = npc.ma.median(lbvi_times_masked, axis=0)
    lbvi_times_masked = npc.ma.filled(lbvi_times_masked, np.nan) # fill masked values with nan to then use nanquantile
    lbvi_times_err = np.vstack((lbvi_times_median - np.nanquantile(lbvi_times_masked, 0.25, axis=0), np.nanquantile(lbvi_times_masked, 0.75, axis=0) - lbvi_times_median))
    lbvi_times_err2 = np.vstack((lbvi_times_median - np.nanquantile(lbvi_times_masked, 0.05, axis=0), np.nanquantile(lbvi_times_masked, 0.95, axis=0) - lbvi_times_median))

    lbvi_kernels_masked = npc.ma.masked_where(lbvi_kernels == 0, lbvi_kernels) # mask 0's
    lbvi_kernels_median = npc.ma.median(lbvi_kernels_masked, axis=0)
    lbvi_kernels_masked = npc.ma.filled(lbvi_kernels_masked, np.nan) # fill masked values with nan to then use nanquantile
    lbvi_kernels_err = np.vstack((lbvi_kernels_median - np.nanquantile(lbvi_kernels_masked, 0.25, axis=0), np.nanquantile(lbvi_kernels_masked, 0.75, axis=0) - lbvi_kernels_median))
    lbvi_kernels_err2 = np.vstack((lbvi_kernels_median - np.nanquantile(lbvi_kernels_masked, 0.05, axis=0), np.nanquantile(lbvi_kernels_masked, 0.95, axis=0) - lbvi_kernels_median))

    lbvi_ksd_masked = npc.ma.masked_where(lbvi_ksd == 0, lbvi_ksd) # mask 0's
    lbvi_ksd_median = npc.ma.median(lbvi_ksd_masked, axis=0)
    lbvi_ksd_masked = npc.ma.filled(lbvi_ksd_masked, np.nan) # fill masked values with nan to then use nanquantile
    lbvi_ksd_err = np.vstack((lbvi_ksd_median - np.nanquantile(lbvi_ksd_masked, 0.25, axis=0), np.nanquantile(lbvi_ksd_masked, 0.75, axis=0) - lbvi_ksd_median))
    lbvi_ksd_err2 = np.vstack((lbvi_ksd_median - np.nanquantile(lbvi_ksd_masked, 0.05, axis=0), np.nanquantile(lbvi_ksd_masked, 0.95, axis=0) - lbvi_ksd_median))

    # UBVI
    ubvi_times_masked = npc.ma.masked_where(ubvi_times == 0, ubvi_times) # mask 0's
    ubvi_times_median = npc.ma.median(ubvi_times_masked, axis=0)
    ubvi_times_masked = npc.ma.filled(ubvi_times_masked, np.nan) # fill masked values with nan to then use nanquantile
    ubvi_times_err = np.vstack((ubvi_times_median - np.nanquantile(ubvi_times_masked, 0.25, axis=0), np.nanquantile(ubvi_times_masked, 0.75, axis=0) - ubvi_times_median))
    ubvi_times_err2 = np.vstack((ubvi_times_median - np.nanquantile(ubvi_times_masked, 0.05, axis=0), np.nanquantile(ubvi_times_masked, 0.95, axis=0) - ubvi_times_median))

    ubvi_kernels_masked = npc.ma.masked_where(ubvi_kernels == 0, ubvi_kernels) # mask 0's
    ubvi_kernels_median = npc.ma.median(ubvi_kernels_masked, axis=0)
    ubvi_kernels_masked = npc.ma.filled(ubvi_kernels_masked, np.nan) # fill masked values with nan to then use nanquantile
    ubvi_kernels_err = np.vstack((ubvi_kernels_median - np.nanquantile(ubvi_kernels_masked, 0.25, axis=0), np.nanquantile(ubvi_kernels_masked, 0.75, axis=0) - ubvi_kernels_median))
    ubvi_kernels_err2 = np.vstack((ubvi_kernels_median - np.nanquantile(ubvi_kernels_masked, 0.05, axis=0), np.nanquantile(ubvi_kernels_masked, 0.95, axis=0) - ubvi_kernels_median))

    ubvi_ksd_masked = npc.ma.masked_where(ubvi_ksd == 0, ubvi_ksd) # mask 0's
    ubvi_ksd_median = npc.ma.median(ubvi_ksd_masked, axis=0)
    ubvi_ksd_masked = npc.ma.filled(ubvi_ksd_masked, np.nan) # fill masked values with nan to then use nanquantile
    ubvi_ksd_err = np.vstack((ubvi_ksd_median - np.nanquantile(ubvi_ksd_masked, 0.25, axis=0), np.nanquantile(ubvi_ksd_masked, 0.75, axis=0) - ubvi_ksd_median))
    ubvi_ksd_err2 = np.vstack((ubvi_ksd_median - np.nanquantile(ubvi_ksd_masked, 0.05, axis=0), np.nanquantile(ubvi_ksd_masked, 0.95, axis=0) - ubvi_ksd_median))

    # BVI
    bvi_times_masked = npc.ma.masked_where(bvi_times == 0, bvi_times) # mask 0's
    bvi_times_median = npc.ma.median(bvi_times_masked, axis=0)
    bvi_times_masked = npc.ma.filled(bvi_times_masked, np.nan) # fill masked values with nan to then use nanquantile
    bvi_times_err = np.vstack((bvi_times_median - np.nanquantile(bvi_times_masked, 0.25, axis=0), np.nanquantile(bvi_times_masked, 0.75, axis=0) - bvi_times_median))
    bvi_times_err2 = np.vstack((bvi_times_median - np.nanquantile(bvi_times_masked, 0.05, axis=0), np.nanquantile(bvi_times_masked, 0.95, axis=0) - bvi_times_median))

    bvi_kernels_masked = npc.ma.masked_where(bvi_kernels == 0, bvi_kernels) # mask 0's
    bvi_kernels_median = npc.ma.median(bvi_kernels_masked, axis=0)
    bvi_kernels_masked = npc.ma.filled(bvi_kernels_masked, np.nan) # fill masked values with nan to then use nanquantile
    bvi_kernels_err = np.vstack((bvi_kernels_median - np.nanquantile(bvi_kernels_masked, 0.25, axis=0), np.nanquantile(bvi_kernels_masked, 0.75, axis=0) - bvi_kernels_median))
    bvi_kernels_err2 = np.vstack((bvi_kernels_median - np.nanquantile(bvi_kernels_masked, 0.05, axis=0), np.nanquantile(bvi_kernels_masked, 0.95, axis=0) - bvi_kernels_median))

    bvi_ksd_masked = npc.ma.masked_where(bvi_ksd == 0, bvi_ksd) # mask 0's
    bvi_ksd_median = npc.ma.median(bvi_ksd_masked, axis=0)
    bvi_ksd_masked = npc.ma.filled(bvi_ksd_masked, np.nan) # fill masked values with nan to then use nanquantile
    bvi_ksd_err = np.vstack((bvi_ksd_median - np.nanquantile(bvi_ksd_masked, 0.25, axis=0), np.nanquantile(bvi_ksd_masked, 0.75, axis=0) - bvi_ksd_median))
    bvi_ksd_err2 = np.vstack((bvi_ksd_median - np.nanquantile(bvi_ksd_masked, 0.05, axis=0), np.nanquantile(bvi_ksd_masked, 0.95, axis=0) - bvi_ksd_median))

    # plot
    niter = 30
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ebalpha = 0.75
    ebalpha2 = 0.5

    # times error bars
    #ax1.errorbar(range(1,niter+1), lbvi_times_median[:niter], yerr = lbvi_times_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
    #ax1.errorbar(range(1,niter+1), ubvi_times_median[:niter], yerr = ubvi_times_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
    #ax1.errorbar(range(1,niter+1), bvi_times_median[:niter], yerr = bvi_times_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
    ax1.errorbar(range(1,niter+1), lbvi_times_median[:niter], yerr = lbvi_times_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha)
    ax1.errorbar(range(1,niter+1), ubvi_times_median[:niter], yerr = ubvi_times_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha)
    ax1.errorbar(range(1,niter+1), bvi_times_median[:niter], yerr = bvi_times_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha)
    ax1.plot(range(1,niter+1), lbvi_times_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI')
    ax1.plot(range(1,niter+1), ubvi_times_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI')
    ax1.plot(range(1,niter+1), bvi_times_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI')

    # kernels error bars
    #ax2.errorbar(range(1,niter+1), lbvi_kernels_median[:niter], yerr = lbvi_kernels_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
    #ax2.errorbar(range(1,niter+1), ubvi_kernels_median[:niter], yerr = ubvi_kernels_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
    #ax2.errorbar(range(1,niter+1), bvi_kernels_median[:niter], yerr = bvi_kernels_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
    ax2.errorbar(range(1,niter+1), lbvi_kernels_median[:niter], yerr = lbvi_kernels_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha)
    ax2.errorbar(range(1,niter+1), ubvi_kernels_median[:niter], yerr = ubvi_kernels_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha)
    ax2.errorbar(range(1,niter+1), bvi_kernels_median[:niter], yerr = bvi_kernels_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha)
    ax2.plot(range(1,niter+1), lbvi_kernels_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI')
    ax2.plot(range(1,niter+1), ubvi_kernels_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI')
    ax2.plot(range(1,niter+1), bvi_kernels_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI')

    # ksd error bars
    #ax3.errorbar(range(1,niter+1), lbvi_ksd_median[:niter], yerr = lbvi_ksd_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
    #ax3.errorbar(range(1,niter+1), ubvi_ksd_median[:niter], yerr = ubvi_ksd_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
    #ax3.errorbar(range(1,niter+1), bvi_ksd_median[:niter], yerr = bvi_ksd_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
    ax3.errorbar(range(1,niter+1), lbvi_ksd_median[:niter], yerr = lbvi_ksd_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha)
    ax3.errorbar(range(1,niter+1), ubvi_ksd_median[:niter], yerr = ubvi_ksd_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha)
    ax3.errorbar(range(1,niter+1), bvi_ksd_median[:niter], yerr = bvi_ksd_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha)
    ax3.plot(range(1,niter+1), lbvi_ksd_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI')
    ax3.plot(range(1,niter+1), ubvi_ksd_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI')
    ax3.plot(range(1,niter+1), bvi_ksd_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI')

    # add labels and save
    #ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('CPU time (s)')
    ax1.legend(fontsize = 'xx-small', loc = 'upper left')
    #ax2.set_xlabel('Iteration #')
    ax2.set_ylabel('# of kernels')

    ax3.set_xlabel('Iteration #')
    ax3.set_ylabel('KSD')
    ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig(path + 'times_kernels_newv2.' + extension, dpi=900, bbox_inches='tight')
###################



print('done plotting!')
