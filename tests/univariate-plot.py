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
import lbvi_smc
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
parser.add_argument('--lbvi_smc', action = "store_true",
help = 'plot lbvi with smc components?')
parser.add_argument('--smc', type = str, default = 'smc', choices=['smc'],
help = 'smc sampler to use in the lbvi mixture')
parser.add_argument('--smc_eps', type = float, default = 0.01,
help = 'step size of the smc discretization')
parser.add_argument('--smc_sd', type = float, default = 1.,
help = 'std deviation of the rwmh rejuvenation kernel in smc')
parser.add_argument('--smc_T', type = int, default = 1,
help = 'number of steps of the rwmh rejuvenation kernel in smc')
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
lbvi_smc_flag = args.lbvi_smc
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

# import smc sampler
smc_kernel = args.smc
smc_eps = args.smc_eps
smc_sd = args.smc_sd
smc_T = args.smc_T
if smc_kernel == 'smc':
    from smc.smc import *
    smc = create_smc(sd = smc_sd, steps = smc_T)

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
lbvi_smc_color = '#39558CFF'
lbvi_color = '#0756f0FF'
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

            if lbvi_smc_flag:
                # retrieve lbvi smc results
                tmp_path = inpath + 'lbvi_smc/'
                y = np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy')
                w = np.load(tmp_path + 'w_' + str(r+1) + '_' + str(tol) + '.npy')
                betas = np.load(tmp_path + 'betas_' + str(r+1) + '_' + str(tol) + '.npy')
                beta_ls = [np.load(tmp_path +'beta_ls_' + str(r+1) + '_' + str(tol) + '/beta_ls_' + str(n+1) + '.npy') for n in range(y.shape[0])]
                lbvi_smc_logq = lambda x : lbvi_smc.mix_logpdf(x, logp, y, w, smc, smc_sd, betas, beta_ls, 10000, None)

            if lbvi_flag:
                # retrieve lbvi settings
                tmp_path = inpath + 'lbvi/'
                y = np.load(tmp_path + 'y_' + str(r+1) + '_' + str(tol) + '.npy')
                w = np.load(tmp_path + 'w_' + str(r+1) + '_' + str(tol) + '.npy')
                T = np.load(tmp_path + 'T_' + str(r+1) + '_' + str(tol) + '.npy')
                lbvi_sample = lbvi.mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = 1)

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
            t = np.linspace(-40, 40, 2000)
            f = logp(t[:,np.newaxis])
            plt.plot(t, f, linestyle = 'solid', color = 'black', label = 'log p(x)', lw = normal_linewidth)

            if lbvi_smc_flag:
                # add lbvi smc log density
                plt.plot(t, lbvi_smc_logq(t[:,np.newaxis]), linestyle = 'dashed', color = lbvi_smc_color, label='LBVI', lw = normal_linewidth)

            if lbvi_flag:
                # add lbvi log density based on kde
                lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample), bw_method = 0.25).evaluate(t)
                plt.plot(t, np.log(lbvi_kde), linestyle = 'dashed', color = lbvi_color, label = 'LBVI MCMC', lw = normal_linewidth)

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
            plt.xlim(-40,40)
            plt.ylim(-20,5)
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

            if lbvi_smc_flag:
                # add lbvi smc density
                plt.plot(t, np.exp(lbvi_smc_logq(t[:,np.newaxis])), linestyle = 'dashed', color = lbvi_smc_color, label='LBVI', lw = normal_linewidth)

            if lbvi_flag:
                # add lbvi histogram
                lbvi_kde = stats.gaussian_kde(np.squeeze(lbvi_sample), bw_method = 0.05).evaluate(t)
                plt.plot(t, lbvi_kde, linestyle = 'dashed', color = lbvi_color, label = 'LBVI MCMC', lw = normal_linewidth)

            if ubvi_flag:
                # add ubvi density
                lq = ubvi.mixture_logpdf(t[:, np.newaxis], ubvi_mu, ubvi_Sig, ubvi_wt)
                plt.plot(t, np.exp(lq), linestyle = 'dashed', color = ubvi_color, label='UBVI', lw = normal_linewidth)

            if bvi_flag:
                # add bvi density
                plt.plot(t, np.exp(bvi_logq(t[:,np.newaxis])), linestyle = 'dashed', color = bvi_color, label='BBBVI', lw = normal_linewidth)

            if gvi_flag:
                # add gvi density
                plt.plot(t, np.exp(gvi_logq(t[:,np.newaxis])), linestyle = 'dashed', color = gvi_color, label='GVI', alpha = muted_alpha, lw = muted_linewidth)

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

            # add labels
            #plt.xlabel('x')
            #plt.ylabel('Density')
            #plt.title('Density comparison')
            plt.xlim(xlim)
            plt.legend(fontsize = legend_fontsize, frameon = False, loc = 'upper left')

            # save plot
            plt.savefig(path + 'densities/density_comparison'  + str(r+1) + '_' + str(tol) + '.' + extension, dpi=900, bbox_inches='tight')
            plt.clf()
##########################


# TIMES AND KERNELS TOGETHER ###################
if lbvi_smc_flag and bvi_flag and ubvi_flag:

    # init arrays
    ttols = tols.shape[0]
    niter = np.load(inpath + 'lbvi_smc/cput_1_' + str(tols[0]) + '.npy').shape[0]-1
    lbvi_times = np.zeros((reps*ttols, niter))
    lbvi_kernels = np.zeros((reps*ttols, niter))
    lbvi_kl = np.zeros((reps*ttols, niter))

    ubvi_times = np.zeros((reps*ttols, niter))
    ubvi_kernels = np.zeros((reps*ttols, niter))
    ubvi_kl = np.zeros((reps*ttols, niter))

    bvi_times = np.zeros((reps*ttols, niter))
    bvi_kernels = np.zeros((reps*ttols, niter))
    bvi_kl = np.zeros((reps*ttols, niter))

    # populate arrays
    counter = 0
    for r in range(reps):
        for t in range(tols.shape[0]):
            cput = np.load(inpath + 'lbvi_smc/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            tmp_n = cput.shape[0]#+1 # sometimes there will be fewer than niter iterations, so get how many there were and substitute
            lbvi_times[counter,:tmp_n] = cput
            lbvi_kernels[counter,:tmp_n] = np.load(inpath + 'lbvi_smc/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            tmp_kl = np.load(inpath + 'lbvi_smc/kl_' + str(r+1) + '_' + str(tols[t]) + '.npy')[1:]
            tmp_kl[tmp_kl == np.NINF] = 0. # no worries, further down the road we mask 0's anyway, so this get's masked
            lbvi_kl[counter,:tmp_n] = tmp_kl

            cput = np.load(inpath + 'ubvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_n = cput.shape[0]
            ubvi_times[counter,:tmp_n] = cput
            ubvi_kernels[counter,:tmp_n] = np.load(inpath + 'ubvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl = np.load(inpath + 'ubvi/kl_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl[tmp_kl == np.NINF] = 0. # no worries, further down the road we mask 0's anyway, so this get's masked
            ubvi_kl[counter,:tmp_n] = tmp_kl

            cput = np.load(inpath + 'bvi/cput_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_n = cput.shape[0]
            bvi_times[counter,:tmp_n] = cput
            bvi_kernels[counter,:tmp_n] = np.load(inpath + 'bvi/kernels_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl = np.load(inpath + 'bvi/kl_' + str(r+1) + '_' + str(tols[t]) + '.npy')
            tmp_kl[tmp_kl == np.NINF] = 0. # no worries, further down the road we mask 0's anyway, so this get's masked
            bvi_kl[counter,:tmp_n] = tmp_kl

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

    bvi_kl_masked = npc.ma.masked_where(bvi_kl == 0, bvi_kl) # mask 0's
    bvi_kl_median = npc.ma.median(bvi_kl_masked, axis=0)
    bvi_kl_masked = npc.ma.filled(bvi_kl_masked, np.nan) # fill masked values with nan to then use nanquantile
    bvi_kl_err = np.vstack((bvi_kl_median - np.nanquantile(bvi_kl_masked, 0.25, axis=0), np.nanquantile(bvi_kl_masked, 0.75, axis=0) - bvi_kl_median))
    bvi_kl_err2 = np.vstack((bvi_kl_median - np.nanquantile(bvi_kl_masked, 0.05, axis=0), np.nanquantile(bvi_kl_masked, 0.95, axis=0) - bvi_kl_median))

    # plot
    #niter = 30
    plt.rcParams["figure.figsize"] = (15,7.5)
    plt.rcParams.update({'font.size': 22})
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ebalpha = 0.75
    ebalpha2 = 0.5
    big_lw = 4

    # times error bars
    #ax1.errorbar(range(1,niter+1), lbvi_times_median[:niter], yerr = lbvi_times_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
    #ax1.errorbar(range(1,niter+1), ubvi_times_median[:niter], yerr = ubvi_times_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
    #ax1.errorbar(range(1,niter+1), bvi_times_median[:niter], yerr = bvi_times_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
    ax1.errorbar(range(1,niter+1), lbvi_times_median[:niter], yerr = lbvi_times_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha, lw = big_lw)
    ax1.errorbar(range(1,niter+1), ubvi_times_median[:niter], yerr = ubvi_times_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha, lw = big_lw)
    ax1.errorbar(range(1,niter+1), bvi_times_median[:niter], yerr = bvi_times_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha)
    ax1.plot(range(1,niter+1), lbvi_times_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI', lw = big_lw)
    ax1.plot(range(1,niter+1), ubvi_times_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI', lw = big_lw)
    ax1.plot(range(1,niter+1), bvi_times_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI', lw = big_lw)

    # kernels error bars
    #ax2.errorbar(range(1,niter+1), lbvi_kernels_median[:niter], yerr = lbvi_kernels_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
    #ax2.errorbar(range(1,niter+1), ubvi_kernels_median[:niter], yerr = ubvi_kernels_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
    #ax2.errorbar(range(1,niter+1), bvi_kernels_median[:niter], yerr = bvi_kernels_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
    ax2.errorbar(range(1,niter+1), lbvi_kernels_median[:niter], yerr = lbvi_kernels_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha, lw = big_lw)
    ax2.errorbar(range(1,niter+1), ubvi_kernels_median[:niter], yerr = ubvi_kernels_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha, lw = big_lw)
    ax2.errorbar(range(1,niter+1), bvi_kernels_median[:niter], yerr = bvi_kernels_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha, lw = big_lw)
    ax2.plot(range(1,niter+1), lbvi_kernels_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI', lw = big_lw)
    ax2.plot(range(1,niter+1), ubvi_kernels_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI', lw = big_lw)
    ax2.plot(range(1,niter+1), bvi_kernels_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI', lw = big_lw)

    # kl error bars
    #ax3.errorbar(range(1,niter+1), lbvi_kl_median[:niter], yerr = lbvi_kl_err2[:,:niter], linestyle = 'dotted', color = lbvi_color, alpha = ebalpha2)
    #ax3.errorbar(range(1,niter+1), ubvi_kl_median[:niter], yerr = ubvi_kl_err2[:,:niter], linestyle = 'dotted', color = ubvi_color, alpha = ebalpha2)
    #ax3.errorbar(range(1,niter+1), bvi_kl_median[:niter], yerr = bvi_kl_err2[:,:niter], linestyle = 'dotted', color = bvi_color, alpha = ebalpha2)
    ax3.errorbar(range(1,niter+1), lbvi_kl_median[:niter], yerr = lbvi_kl_err[:,:niter], linestyle = 'solid', color = lbvi_color, alpha = ebalpha, lw = big_lw)
    ax3.errorbar(range(1,niter+1), ubvi_kl_median[:niter], yerr = ubvi_kl_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha, lw = big_lw)
    ax3.errorbar(range(1,niter+1), bvi_kl_median[:niter], yerr = bvi_kl_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha, lw = big_lw)
    ax3.plot(range(1,niter+1), lbvi_kl_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI', lw = big_lw)
    ax3.plot(range(1,niter+1), ubvi_kl_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI', lw = big_lw)
    ax3.plot(range(1,niter+1), bvi_kl_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI', lw = big_lw)

    # add labels and save
    #ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('CPU time (s)')
    ax1.set_yscale('log')
    #ax1.legend(fontsize = 'xx-small', loc = 'upper left')
    #ax2.set_xlabel('Iteration #')
    ax2.set_ylabel('# of kernels')

    ax3.set_xlabel('Iteration #')
    ax3.set_ylabel('KL estimate')

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
plt.errorbar(range(1,niter+1), ubvi_kl_median[:niter], yerr = ubvi_kl_err[:,:niter], linestyle = 'solid', color = ubvi_color, alpha = ebalpha)
plt.errorbar(range(1,niter+1), bvi_kl_median[:niter], yerr = bvi_kl_err[:,:niter], linestyle = 'solid', color = bvi_color, alpha = ebalpha)
plt.plot(range(1,niter+1), lbvi_kl_median[:niter], linestyle = 'solid', color = lbvi_color, label='LBVI')
plt.plot(range(1,niter+1), ubvi_kl_median[:niter], linestyle = 'solid', color = ubvi_color, label='UBVI')
plt.plot(range(1,niter+1), bvi_kl_median[:niter], linestyle = 'solid', color = bvi_color, label='BVI')

# add labels and save

plt.xlabel('Iteration #')
plt.ylabel('KL estimate')
#ax3.set_yscale('log')

plt.tight_layout()
plt.savefig(path + 'kl.' + extension, dpi=900, bbox_inches='tight')
###################

print('done plotting!')
