# plot the results of univariate simulation stored in .cvs files in given folder

# PREAMBLE ####
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import argparse
import sys, os
import imageio

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../lbvi/'))
import lbvi # functions to do locally-adapted boosting variational inference

# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="plot results from locally-adapted boosting variational inference")

parser.add_argument('--inpath', type = str, default = 'results/',
help = 'path of folder where csv files are stored')
parser.add_argument('--outpath', type = str, default = 'results/plots/',
help = 'path of folder where plots will be saved')
parser.add_argument('--target', type = str, default = '4-mixture', choices=['4-mixture', 'cauchy'],
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
    xlim = np.array([-6, 6]) # for plotting
if target == 'cauchy':
    from targets.cauchy import *
    xlim = np.array([-10, 10])


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
up = lbvi.up_gen(kernel, sp, dk_x, dk_y, dk_xy)

# READ DATA FRAMES IN PATH ####
metadata = pd.DataFrame({'file_name': [], 'N': []})

for file_name in glob.glob(inpath + 'results' + '*.csv'):
    # read file and save info
    dat = pd.read_csv(file_name)
    metadata = metadata.append(pd.DataFrame({'file_name': [file_name], 'N': [len(dat.index)]}))
# end for


# PLOT ####
print('begin plotting!')


# gif
# code from https://stackoverflow.com/questions/41228209/making-gif-from-images-using-imageio-in-python
jpg_dir = inpath + 'plots/'
jpg_dir = os.listdir(jpg_dir)
jpg_dir = np.setdiff1d(jpg_dir, 'weight_trace')
number = np.zeros(len(jpg_dir))
i = 0
# fix names
for x in jpg_dir:
    x = x[5:]
    x = x[:-4]
    number[i] = int(x)
    i = i+1
# end for


db = pd.DataFrame({'file_name': jpg_dir,
                   'number': number})
db = db.sort_values(by=['number'])

images = []
for file_name in db.file_name:
    images.append(imageio.imread(inpath + 'plots/' + file_name))
imageio.mimsave(path + 'evolution.gif', images, fps = 4)
# gif done



ss = metadata.N.unique() # save sample sizes
errors = pd.DataFrame({'N': [], 'disc': []})

# log densities
for N in ss:
    # files with that sample size
    files = metadata[metadata.N == N].file_name

    # initialize plot with target log density
    t = np.linspace(xlim[0], xlim[1], 2000)
    f = logp(t)
    plt.plot(t, f, 'k-', label = 'target', linewidth = 1, markersize = 1.5)
    i = 1

    for file in files:
        # read file and generate resulting approximation
        dat = pd.read_csv(file)
        w = np.array(dat.w)
        T =  np.array(dat.steps)
        dat = dat.drop(['w', 'steps'], axis = 1)
        x = np.squeeze(np.array(dat))
        kk = lbvi.mix_sample(10000, y = x, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
        yy = stats.gaussian_kde(kk, bw_method = 0.05).evaluate(t)

        # plot log density estimation
        if i == 1: plt.plot(t, np.log(yy), '--b', label = 'approximation')
        else: plt.plot(t, np.log(yy), '--b')
        i += 1

        # calculate discrepancy for future plotting
        errors = errors.append(pd.DataFrame({'N': [N], 'disc': lbvi.ksd(logp, x, T, w, up, kernel_sampler, B = 100000)}))

    # end for

    # save plot
    plt.xlabel('x')
    plt.ylabel('log-density')
    plt.title('log-density for mixtures with N = ' + str(N))
    plt.legend()
    title = 'log-density_N' + str(N) + '.'
    plt.savefig(path + title + extension, dpi=900)
    plt.clf()
# end for



# densities
for N in ss:
    # files with that sample size
    files = metadata[metadata.N == N].file_name

    # initialize plot with target log density
    t = np.linspace(xlim[0], xlim[1], 2000)
    f = p(t)
    plt.plot(t, f, 'k-', label = 'target', linewidth = 1, markersize = 1.5)
    i = 1

    for file in files:
        # read file and generate resulting approximation
        dat = pd.read_csv(file)
        w = np.array(dat.w)
        T =  np.array(dat.steps)
        dat = dat.drop(['w', 'steps'], axis = 1)
        x = np.array(dat)
        kk = lbvi.mix_sample(10000, y = x, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)

        # plot log density estimation
        if i == 1: plt.hist(kk, label = 'approximation', density = True, bins = 50)
        else: plt.hist(kk, density = True, bins = 50)
        i += 1


    # end for

    # save plot
    plt.xlabel('x')
    plt.ylabel('density')
    plt.title('density for mixtures with N = ' + str(N))
    plt.legend()
    title = 'density_N' + str(N) + '.'
    plt.savefig(path + title + extension, dpi=900)
    plt.clf()
# end for

# out discrepancy
errors.to_csv(inpath + 'errors.csv', index = False)

# plot discrepancy
fig, ax1 = plt.subplots()
errors.boxplot(column = 'disc', by = 'N', grid = False)
plt.xlabel('sample size N')
plt.ylabel('ksd')
plt.title('discrepancy boxplot')
plt.suptitle('')
plt.savefig(path + 'disc-plot.' + extension, dpi=900, bbox_inches='tight')

print('done plotting!')
