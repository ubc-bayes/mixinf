# plot the results of bivariate simulation stored in .cvs files in given folder

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
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy', '5-mixture', 'double-banana-gaussian', 'banana'],
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
Levels = 4
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
if target == 'double-banana-gaussian':
    from targets.doublebanana_gaussian import *
    xlim = np.array([-25, 25])
    ylim = np.array([-30, 30])
    Levels = 4


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
normal_linewidth = 1.5
muted_alpha = 0.4 # for toning down less important lines
muted_linewidth = 1.5
legend_fontsize = 'medium'

if dens_plots:
    for r in np.arange(reps):
        print(str(r+1) + '/'+ str(reps), end='\r')
        for tol in tols:
            lbvi_flag = args.lbvi
            hmc_flag = args.hmc
            rwmh_flag = args.rwmh

            if lbvi_flag:
                # retrieve lbvi settings
                tmp_path = inpath + 'lbvi/'
                y = np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy')
                w = np.load(tmp_path + 'w_' + str(r+1) + '_' + str(tol) + '.npy')
                T = np.load(tmp_path + 'T_' + str(r+1) + '_' + str(tol) + '.npy')
                lbvi_sample = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = 1)
                lbvi_sample = lbvi_sample[~np.isnan(lbvi_sample).any(axis=-1)]
                lbvi_sample = lbvi_sample[~np.isinf(lbvi_sample).any(axis=-1)]
                if lbvi_sample.size == 0:
                    print('not plotting lbvi')
                    lbvi_flag = False

            if ubvi_flag:
                # retrieve ubvi results
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
                gvi_logq = lambda x : -0.5*2*np.log(2*np.pi) - 0.5*1*np.log(SigmaLogDet) - 0.5*((x-mu).dot(SigmaInv)*(x-mu)).sum(axis=-1)


            if hmc_flag:
                # retrieve hmc sample
                tmp_path = inpath + 'hmc/'
                hmc = np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy')
                hmc = hmc[~np.isnan(hmc).any(axis=-1)]
                hmc = hmc[~np.isinf(hmc).any(axis=-1)]
                if hmc.size == 0:
                    print('not plotting hmc')
                    hmc_flag = False


            if rwmh_flag:
                # retrieve rwmh sample
                tmp_path = inpath + 'rwmh/'
                rwmh = np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy')
                rwmh = rwmh[~np.isnan(rwmh).any(axis=-1)]
                rwmh = rwmh[~np.isinf(rwmh).any(axis=-1)]
                if rwmh.size == 0:
                    print('not plotting rwmh')
                    rwmh_flag = False


            # DENSITY PLOT
            # initialize plot with target density contour
            nn = 100
            xx = np.linspace(xlim[0], xlim[1], nn)
            yy = np.linspace(ylim[0], ylim[1], nn)
            tt = np.array(np.meshgrid(xx, yy)).T.reshape(nn**2, 2)
            lp = logp(tt).reshape(nn, nn).T
            cp = plt.contour(xx, yy, np.exp(lp), levels = Levels, colors = 'black', linewidths = normal_linewidth)
            hcp,_ = cp.legend_elements()
            hcps = [hcp[0]]
            legends = ['p(x)']

            if lbvi_flag:
                # add lbvi samples
                lbvi_kde = stats.gaussian_kde(lbvi_sample.T, bw_method = 0.05).evaluate(tt.T).reshape(nn, nn).T
                #plt.scatter(kk[:,0], kk[:,1], marker='.', c='b', alpha = 0.4, label = 'LBVI')
                cp_lbvi = plt.contour(xx, yy, lbvi_kde, levels = 8, colors = lbvi_color, linewidths = normal_linewidth)
                hcp_lbvi,_ = cp_lbvi.legend_elements()
                hcps.append(hcp_lbvi[0])
                legends.append('LBVI')

            if ubvi_flag:
                # add ubvi density
                lq_ubvi = ubvi.mixture_logpdf(tt, ubvi_mu, ubvi_Sig, ubvi_wt).reshape(nn, nn).T
                cp_ubvi = plt.contour(xx, yy, np.exp(lq_ubvi), levels = Levels, colors = ubvi_color, linewidths = normal_linewidth)
                hcp_ubvi,_ = cp_ubvi.legend_elements()
                hcps.append(hcp_ubvi[0])
                legends.append('UBVI')

            if bvi_flag:
                # add bvi density
                lq_bvi = np.exp(bvi_logq(tt)).reshape(nn, nn).T
                cp_bvi = plt.contour(xx, yy, np.exp(lq_bvi), levels = Levels, colors = bvi_color, linewidths = normal_linewidth)
                hcp_bvi,_ = cp_bvi.legend_elements()
                hcps.append(hcp_bvi[0])
                legends.append('BVI')


            if gvi_flag:
                # add gvi density
                lq_gvi = np.exp(gvi_logq(tt)).reshape(nn, nn).T
                cp_gvi = plt.contour(xx, yy, np.exp(lq_gvi), levels = Levels, colors = gvi_color, alpha = muted_alpha, linewidths = muted_linewidth)
                hcp_gvi,_ = cp_gvi.legend_elements()
                hcps.append(hcp_gvi[0])
                legends.append('GVI')


            if rwmh_flag:
                # add rwmh kde density
                rwmh_kde = stats.gaussian_kde(rwmh.T, bw_method = 0.15).evaluate(tt.T).reshape(nn, nn).T
                cp_rwmh = plt.contour(xx, yy, rwmh_kde, linestyles = 'dotted', levels = Levels, colors = rwmh_color, alpha = muted_alpha, linewidths = muted_linewidth)
                hcp_rwmh,_ = cp_rwmh.legend_elements()
                hcps.append(hcp_rwmh[0])
                legends.append('RWMH')

            if hmc_flag:
                # add hmc kde density
                hmc_kde = stats.gaussian_kde(hmc.T, bw_method = 0.2).evaluate(tt.T).reshape(nn, nn).T
                cp_hmc = plt.contour(xx, yy, hmc_kde, linestyles = 'dashdot', levels = Levels, colors = hmc_color, alpha = muted_alpha, linewidths = muted_linewidth)
                #plt.scatter(hmc[:,0], hmc[:,1], color = hmc_color)
                hcp_hmc,_ = cp_hmc.legend_elements()
                hcps.append(hcp_hmc[0])
                legends.append('HMC')


            # add labels
            plt.xlim(xlim)
            plt.ylim(ylim)
            #plt.legend(hcps, legends, fontsize = legend_fontsize, frameon = False, ncol=3)#len(legends)

            # save plot
            plt.savefig(path + 'densities/density_comparison'  + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
            plt.clf()
            ##########################

            ## LOG DENSITY PLOT separate
            #for i in [0,1]:
            #    # initialize plot with target log density
            #    if i == 0: t = np.linspace(xlim[0], xlim[1], nn)
            #    if i == 1: t = np.linspace(ylim[0], ylim[1], nn)
            #    plt.plot(t, ubvi.logsumexp(lp, axis = 1-i), linestyle = 'solid', color = 'black', label = 'log p(x)', lw = normal_linewidth)

            #    if lbvi_flag:
            #        # add lbvi log density based on kde
            #        #lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample[:,i]), bw_method = 0.25).evaluate(t)
            #        plt.plot(t, ubvi.logsumexp(np.log(lbvi_kde), axis = 1-i), linestyle = 'dashed', color = lbvi_color, label = 'LBVI', lw = normal_linewidth)

            #    if ubvi_flag:
            #        # add ubvi log density
            #        plt.plot(t, ubvi.logsumexp(lq_ubvi, axis=1-i), linestyle = 'dashed', color = ubvi_color, label='UBVI', lw = normal_linewidth)

            #    if bvi_flag:
            #        # add bvi log density
            #        plt.plot(t, ubvi.logsumexp(lq_bvi, axis=1-i), linestyle = 'dashed', color = bvi_color, label='BBBVI', lw = normal_linewidth)

            #    if gvi_flag:
            #        # add gvi log density
            #        plt.plot(t, ubvi.logsumexp(lq_gvi, axis=1-i), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)

            #    if hmc_flag:
            #        # add rwmh log density based on kde
            #        #hmc_kde = stats.gaussian_kde(np.squeeze(hmc[:,i]), bw_method = 1).evaluate(t)
            #        plt.plot(t, ubvi.logsumexp(np.log(hmc_kde), axis = 1-i), linestyle = 'dashdot', color = hmc_color, label = 'HMC', alpha = muted_alpha, lw =    #muted_linewidth)

            #    if rwmh_flag:
            #        # add rwmh log density based on kde
            #        #rwmh_kde = stats.gaussian_kde(np.squeeze(rwmh[:,i]), bw_method = 0.15).evaluate(t)
            #        plt.plot(t, ubvi.logsumexp(np.log(rwmh_kde), axis = 1-i), linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw =  #muted_linewidth)

            #    # add labels and save plot
            #    if i == 0:
            #        title = 'logdensities/log-density_comparison_xaxis'
            #        plt.xlim(xlim)
            #    if i == 1:
            #        title = 'logdensities/log-density_comparison_yaxis'
            #        plt.xlim(ylim)
            #    plt.legend(fontsize = legend_fontsize)
            #    plt.savefig(path + title + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
            #    plt.clf()
            ############################

            # LOG DENSITY PLOT (together)
            # initialize plot with target log density
            t1 = np.linspace(xlim[0], xlim[1], nn)
            t2 = np.linspace(ylim[0], ylim[1], nn)

            fig, (ax1, ax2) = plt.subplots(2)
            ax1.plot(t1, ubvi.logsumexp(lp, axis = 1), linestyle = 'solid', color = 'black', label = 'log p(x)', lw = normal_linewidth)
            ax2.plot(t2, ubvi.logsumexp(lp, axis = 0), linestyle = 'solid', color = 'black', label = 'log p(x)', lw = normal_linewidth)

            if lbvi_flag:
                # add lbvi log density based on kde
                ax1.plot(t1, ubvi.logsumexp(np.log(lbvi_kde), axis = 1), linestyle = 'dashed', color = lbvi_color, label = 'LBVI', lw = normal_linewidth)
                ax2.plot(t2, ubvi.logsumexp(np.log(lbvi_kde), axis = 0), linestyle = 'dashed', color = lbvi_color, label = 'LBVI', lw = normal_linewidth)

            if ubvi_flag:
                # add ubvi log density
                ax1.plot(t1, ubvi.logsumexp(lq_ubvi, axis=1), linestyle = 'dashed', color = ubvi_color, label='UBVI', lw = normal_linewidth)
                ax2.plot(t2, ubvi.logsumexp(lq_ubvi, axis=0), linestyle = 'dashed', color = ubvi_color, label='UBVI', lw = normal_linewidth)

            if bvi_flag:
                # add bvi log density
                ax1.plot(t1, ubvi.logsumexp(lq_bvi, axis=1), linestyle = 'dashed', color = bvi_color, label='BBBVI', lw = normal_linewidth)
                ax2.plot(t2, ubvi.logsumexp(lq_bvi, axis=0), linestyle = 'dashed', color = bvi_color, label='BBBVI', lw = normal_linewidth)

            if gvi_flag:
                # add gvi log density
                ax1.plot(t1, ubvi.logsumexp(lq_gvi, axis=1), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)
                ax2.plot(t2, ubvi.logsumexp(lq_gvi, axis=0), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)

            if hmc_flag:
                # add rwmh log density based on kde
                ax1.plot(t1, ubvi.logsumexp(np.log(hmc_kde), axis = 1), linestyle = 'dashdot', color = hmc_color, label = 'HMC', alpha = muted_alpha, lw = muted_linewidth)
                ax2.plot(t2, ubvi.logsumexp(np.log(hmc_kde), axis = 0), linestyle = 'dashdot', color = hmc_color, label = 'HMC', alpha = muted_alpha, lw = muted_linewidth)

            if rwmh_flag:
                # add rwmh log density based on kde
                ax1.plot(t1, ubvi.logsumexp(np.log(rwmh_kde),axis = 1), linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw =    muted_linewidth)
                ax2.plot(t2, ubvi.logsumexp(np.log(rwmh_kde),axis = 0), linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw  = muted_linewidth)

            # add limits and save plot
            ax1.set_xlim(xlim)
            ax1.set_ylim(-20,5)
            ax2.set_xlim(-20,20)
            ax2.set_ylim(-20,5)
            plt.tight_layout()
            plt.savefig(path + 'logdensities/log-density_comparison' + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
            plt.clf()

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
    niter = 20
    lbvi_times = np.zeros((reps*ttols, niter))
    lbvi_kernels = np.zeros((reps*ttols, niter))
    lbvi_ksd = np.zeros((reps*ttols, niter))
    lbvi_kl = np.zeros((reps*ttols, niter))

    ubvi_times = np.zeros((reps*ttols, niter))
    ubvi_kernels = np.zeros((reps*ttols, niter))
    ubvi_ksd = np.zeros((reps*ttols, niter))
    ubvi_kl = np.zeros((reps*ttols, niter))

    bvi_times = np.zeros((reps*ttols, niter))
    bvi_kernels = np.zeros((reps*ttols, niter))
    bvi_ksd = np.zeros((reps*ttols, niter))
    bvi_kl = np.zeros((reps*ttols, niter))

    # populate arrays
    counter = 0
    for r in range(reps):
        for t in range(tols.shape[0]):
            cput = np.load(inpath + 'lbvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            tmp_n = cput.shape[0]#+1 # sometimes there will be fewer than niter iterations, so get how many there were and substitute
            lbvi_times[counter,:tmp_n] = cput
            lbvi_kernels[counter,:tmp_n] = np.load(inpath + 'lbvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            lbvi_ksd[counter,:tmp_n] = np.load(inpath + 'lbvi/obj_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            tmp_kl = np.load(inpath + 'lbvi/kl_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            tmp_kl[tmp_kl == np.NINF] = 0. # no worries, further down the road we mask 0's anyway, so this get's masked
            lbvi_kl[counter,:tmp_n] = tmp_kl

            cput = np.load(inpath + 'ubvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_n = cput.shape[0]
            ubvi_times[counter,:tmp_n] = cput
            ubvi_kernels[counter,:tmp_n] = np.load(inpath + 'ubvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            ubvi_ksd[counter,:tmp_n] = np.load(inpath + 'ubvi/obj_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl = np.load(inpath + 'ubvi/kl_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl[tmp_kl == np.NINF] = 0. # no worries, further down the road we mask 0's anyway, so this get's masked
            ubvi_kl[counter,:tmp_n] = tmp_kl

            cput = np.load(inpath + 'bvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_n = cput.shape[0]
            bvi_times[counter,:tmp_n] = cput
            bvi_kernels[counter,:tmp_n] = np.load(inpath + 'bvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            bvi_ksd[counter,:tmp_n] = np.load(inpath + 'bvi/obj_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl = np.load(inpath + 'bvi/kl_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl[tmp_kl == np.NINF] = 0. # no worries, further down the road we mask 0's anyway, so this get's masked
            bvi_kl[counter,:tmp_n] = tmp_kl

            counter += 1
        # end for
    # end for

    # make sure kl's are positive by shifting by the min
    #minkl = np.abs(np.amin(np.minimum(lbvi_kl, np.minimum(ubvi_kl, bvi_kl)))) + 1e-100
    #lbvi_kl += minkl
    #ubvi_kl += minkl
    #bvi_kl += minkl


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

    lbvi_kl_masked = npc.ma.masked_where(lbvi_kl == 0, lbvi_kl) # mask 0's
    lbvi_kl_median = npc.ma.median(lbvi_kl_masked, axis=0)
    lbvi_kl_masked = npc.ma.filled(lbvi_kl_masked, np.nan) # fill masked values with nan to then use nanquantile
    lbvi_kl_err = np.vstack((lbvi_kl_median - np.nanquantile(lbvi_kl_masked, 0.25, axis=0), np.nanquantile(lbvi_kl_masked, 0.75, axis=0) - lbvi_kl_median))
    lbvi_kl_err2 = np.vstack((lbvi_kl_median - np.nanquantile(lbvi_kl_masked, 0.05, axis=0), np.nanquantile(lbvi_kl_masked, 0.95, axis=0) - lbvi_kl_median))

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

    ubvi_kl_masked = npc.ma.masked_where(ubvi_kl == 0, ubvi_kl) # mask 0's
    ubvi_kl_median = npc.ma.median(ubvi_kl_masked, axis=0)
    ubvi_kl_masked = npc.ma.filled(ubvi_kl_masked, np.nan) # fill masked values with nan to then use nanquantile
    ubvi_kl_err = np.vstack((ubvi_kl_median - np.nanquantile(ubvi_kl_masked, 0.25, axis=0), np.nanquantile(ubvi_kl_masked, 0.75, axis=0) - ubvi_kl_median))
    ubvi_kl_err2 = np.vstack((ubvi_kl_median - np.nanquantile(ubvi_kl_masked, 0.05, axis=0), np.nanquantile(ubvi_kl_masked, 0.95, axis=0) - ubvi_kl_median))

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


    bvi_kl_masked = npc.ma.masked_where(bvi_kl == 0, bvi_kl) # mask 0's
    bvi_kl_median = npc.ma.median(bvi_kl_masked, axis=0)
    bvi_kl_masked = npc.ma.filled(bvi_kl_masked, np.nan) # fill masked values with nan to then use nanquantile
    bvi_kl_err = np.vstack((bvi_kl_median - np.nanquantile(bvi_kl_masked, 0.25, axis=0), np.nanquantile(bvi_kl_masked, 0.75, axis=0) - bvi_kl_median))
    bvi_kl_err2 = np.vstack((bvi_kl_median - np.nanquantile(bvi_kl_masked, 0.05, axis=0), np.nanquantile(bvi_ksd_masked, 0.95, axis=0) - bvi_kl_median))

    # plot
    #niter = 30
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
    #ax1.legend(fontsize = 'xx-small', loc = 'upper left')
    #ax2.set_xlabel('Iteration #')
    ax2.set_ylabel('# of kernels')

    ax3.set_xlabel('Iteration #')
    ax3.set_ylabel('KSD')
    ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig(path + 'times_kernels.' + extension, dpi=900, bbox_inches='tight')
###################


# TIMES AND KERNELS TOGETHER WITH KL INSTEAD OF KSD ###################
# plot
#niter = 30
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

# kl error bars
#ax3.errorbar(range(1,niter+1), lbvi_kl_median[:niter], yerr = lbvi_kl_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
#ax3.errorbar(range(1,niter+1), ubvi_kl_median[:niter], yerr = ubvi_kl_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
#ax3.errorbar(range(1,niter+1), bvi_kl_median[:niter], yerr = bvi_kl_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
ax3.errorbar(range(1,niter+1), lbvi_kl_median[:niter], yerr = lbvi_kl_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha)
ax3.errorbar(range(1,niter+1), ubvi_kl_median[:niter], yerr = ubvi_kl_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha)
ax3.errorbar(range(1,niter+1), bvi_kl_median[:niter], yerr = bvi_kl_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha)
ax3.plot(range(1,niter+1), lbvi_kl_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI')
ax3.plot(range(1,niter+1), ubvi_kl_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI')
ax3.plot(range(1,niter+1), bvi_kl_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI')

# add labels and save
#ax1.set_xlabel('Iteration #')
ax1.set_ylabel('CPU time (s)')
#ax1.legend(fontsize = 'xx-small', loc = 'upper left')
#ax2.set_xlabel('Iteration #')
ax2.set_ylabel('# of kernels')

ax3.set_xlabel('Iteration #')
ax3.set_ylabel('KL estimate')
#ax3.set_yscale('log')

plt.tight_layout()
plt.savefig(path + 'times_kernels_kl.' + extension, dpi=900, bbox_inches='tight')
###################



# KL ###################
# plot
#niter = 30
plt.clf()
ebalpha = 0.75
ebalpha2 = 0.5

# kl error bars
#plt.errorbar(range(1,niter+1), lbvi_kl_median[:niter], yerr = lbvi_kl_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
#plt.errorbar(range(1,niter+1), ubvi_kl_median[:niter], yerr = ubvi_kl_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
#plt.errorbar(range(1,niter+1), bvi_kl_median[:niter], yerr = bvi_kl_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
plt.errorbar(range(1,niter+1), lbvi_kl_median[:niter], yerr = lbvi_kl_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha)
#plt.errorbar(range(1,niter+1), ubvi_kl_median[:niter], yerr = ubvi_kl_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha)
#plt.errorbar(range(1,niter+1), bvi_kl_median[:niter], yerr = bvi_kl_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha)
plt.plot(range(1,niter+1), lbvi_kl_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI')
#plt.plot(range(1,niter+1), ubvi_kl_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI')
#plt.plot(range(1,niter+1), bvi_kl_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI')

# add labels and save

plt.xlabel('Iteration #')
plt.ylabel('KL estimate')
#ax3.set_yscale('log')

plt.tight_layout()
plt.savefig(path + 'kl.' + extension, dpi=900, bbox_inches='tight')
###################


print('done plotting!')
