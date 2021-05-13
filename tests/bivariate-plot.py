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
parser = argparse.ArgumentParser(description="plot comparison between lbvi and other vi and mcmc routines")

parser.add_argument('--inpath', type = str, default = 'results/',
help = 'path of folder where csv files are stored')
parser.add_argument('--outpath', type = str, default = 'results/plots/',
help = 'path of folder where plots will be saved')
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy', '5-mixture', 'banana-gaussian', 'four-banana'],
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
if target == 'double-banana':
    from targets.double_banana import *
    xlim = np.array([-2.5, 2.5])
    ylim = np.array([-1, 1])
if target == 'banana-gaussian':
    from targets.banana_gaussian import *
    xlim = np.array([-3, 3])
    ylim = np.array([-2, 3])
    Levels = 4
if target == 'four-banana':
    from targets.four_banana import *
    xlim = np.array([-3, 3])
    ylim = np.array([-3, 3])
    Levels = 4


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
gvi_color = '0.2'
hmc_color = '0.3'
rwmh_color = '0.4'
normal_linewidth = 1.5
muted_alpha = 0.4 # for toning down less important lines
muted_linewidth = 1.5
legend_fontsize = 'medium'


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
            lbvi_sample = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
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
            cp_lbvi = plt.contour(xx, yy, lbvi_kde, levels = Levels, colors = lbvi_color, linewidths = normal_linewidth)
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

        # LOG DENSITY PLOT separate
        for i in [0,1]:
            # initialize plot with target log density
            if i == 0: t = np.linspace(xlim[0], xlim[1], nn)
            if i == 1: t = np.linspace(ylim[0], ylim[1], nn)
            plt.plot(t, ubvi.logsumexp(lp, axis = 1-i), linestyle = 'solid', color = 'black', label = 'log p(x)', lw = normal_linewidth)

            if lbvi_flag:
                # add lbvi log density based on kde
                #lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample[:,i]), bw_method = 0.25).evaluate(t)
                plt.plot(t, ubvi.logsumexp(np.log(lbvi_kde), axis = 1-i), linestyle = 'dashed', color = lbvi_color, label = 'LBVI', lw = normal_linewidth)

            if ubvi_flag:
                # add ubvi log density
                plt.plot(t, ubvi.logsumexp(lq_ubvi, axis=1-i), linestyle = 'dashed', color = ubvi_color, label='UBVI', lw = normal_linewidth)

            if bvi_flag:
                # add bvi log density
                plt.plot(t, ubvi.logsumexp(lq_bvi, axis=1-i), linestyle = 'dashed', color = bvi_color, label='BBBVI', lw = normal_linewidth)

            if gvi_flag:
                # add gvi log density
                plt.plot(t, ubvi.logsumexp(lq_gvi, axis=1-i), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)

            if hmc_flag:
                # add rwmh log density based on kde
                #hmc_kde = stats.gaussian_kde(np.squeeze(hmc[:,i]), bw_method = 1).evaluate(t)
                plt.plot(t, ubvi.logsumexp(np.log(hmc_kde), axis = 1-i), linestyle = 'dashdot', color = hmc_color, label = 'HMC', alpha = muted_alpha, lw = muted_linewidth)

            if rwmh_flag:
                # add rwmh log density based on kde
                #rwmh_kde = stats.gaussian_kde(np.squeeze(rwmh[:,i]), bw_method = 0.15).evaluate(t)
                plt.plot(t, ubvi.logsumexp(np.log(rwmh_kde), axis = 1-i), linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw = muted_linewidth)

            # add labels and save plot
            if i == 0:
                title = 'logdensities/log-density_comparison_xaxis'
                plt.xlim(xlim)
            if i == 1:
                title = 'logdensities/log-density_comparison_yaxis'
                plt.xlim(ylim)
            plt.legend(fontsize = legend_fontsize)
            plt.savefig(path + title + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
            plt.clf()
        ###########################

        # LOG DENSITY PLOT v2 (together)
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
            ax1.plot(t1, ubvi.logsumexp(np.log(rwmh_kde),axis = 1), linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw = muted_linewidth)
            ax2.plot(t2, ubvi.logsumexp(np.log(rwmh_kde),axis = 0), linestyle = 'dotted', color = rwmh_color, label = 'RWMH', alpha = muted_alpha, lw = muted_linewidth)

        # add limits and save plot
        ax1.set_xlim(xlim)
        ax2.set_xlim(ylim)
        plt.tight_layout()
        plt.savefig(path + 'logdensities/log-density_comparison' + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
        plt.clf()

msize = 30
pltalpha = 0.75

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



# TIMES AND KERNELS TOGETHER ###################
#plt.rcParams.update({'font.size': 12})
if lbvi_flag and bvi_flag and ubvi_flag:
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2)

    # plot all reps and tols; for the first plot, add labels
    counter = 1
    for r in np.arange(1,reps+1):
        for tol in tols:
            lbvi_times = np.load(inpath + 'lbvi/cput_' + str(r) + '_' + str(tol) + '.npy')[-1]
            lbvi_kernels = np.load(inpath + 'lbvi/kernels_' + str(r) + '_' + str(tol) + '.npy')[-1]
            lbvi_obj = np.load(inpath + 'lbvi/obj_' + str(r) + '_' + str(tol) + '.npy')[-1]
            if counter == 1:
                ax1.scatter(lbvi_times, np.log(lbvi_obj), c = lbvi_color, label = 'LBVI', s = msize, alpha = pltalpha)
            else:
                ax1.scatter(lbvi_times, np.log(lbvi_obj), c = lbvi_color, s = msize, alpha = pltalpha)
            ax2.scatter(lbvi_kernels, np.log(lbvi_obj), c = lbvi_color, s = msize, alpha = pltalpha)

            ubvi_times = np.load(inpath + 'ubvi/cput_' + str(r) + '_' + str(tol) + '.npy')[-1]
            ubvi_kernels = np.load(inpath + 'ubvi/kernels_' + str(r) + '_' + str(tol) + '.npy')[-1]
            ubvi_obj = np.load(inpath + 'ubvi/obj_' + str(r) + '_' + str(tol) + '.npy')[-1]
            if counter == 1:
                ax1.scatter(ubvi_times, np.log(ubvi_obj), c = ubvi_color, label = 'UBVI', s = msize, alpha = pltalpha)
            else:
                ax1.scatter(ubvi_times, np.log(ubvi_obj), c = ubvi_color, s = msize, alpha = pltalpha)
            ax2.scatter(ubvi_kernels, np.log(ubvi_obj), c = ubvi_color, s = msize, alpha = pltalpha)

            bvi_times = np.load(inpath + 'bvi/cput_' + str(r) + '_' + str(tol) + '.npy')[-1]
            bvi_kernels = np.load(inpath + 'bvi/kernels_' + str(r) + '_' + str(tol) + '.npy')[-1]
            bvi_obj = np.load(inpath + 'bvi/obj_' + str(r) + '_' + str(tol) + '.npy')[-1]
            if counter == 1:
                ax1.scatter(bvi_times, np.log(bvi_obj), c = bvi_color, label = 'BVI', s = msize, alpha = pltalpha)
            else:
                ax1.scatter(bvi_times, np.log(bvi_obj), c = bvi_color, s = msize, alpha = pltalpha)
            ax2.scatter(bvi_kernels, np.log(bvi_obj), c = bvi_color, s = msize, alpha = pltalpha)

            counter += 1


    # add labels and save
    ax1.set_xlabel('CPU time (s)')
    ax1.set_ylabel('log KSD')
    #ax1.legend(fontsize = legend_fontsize, loc = 'lower right')
    ax2.set_xlabel('Number of non-zero kernels')
    ax2.set_ylabel('log KSD')
    plt.tight_layout()
    plt.savefig(path + 'times_kernels.' + extension, dpi=900, bbox_inches='tight')
###################



print('done plotting!')
