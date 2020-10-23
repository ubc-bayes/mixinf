# plot the results of bivariate simulation stored in .cvs files in given folder

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

# import the suite of functions from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../mixinf/'))
import nsvmi # functions to do normal seq-opt variational mixture inference

# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="plot results from normal-kernel bivariate variational mixture inference")

parser.add_argument('--inpath', type = str, default = 'results/',
help = 'path of folder where csv files are stored')
parser.add_argument('--outpath', type = str, default = 'results/plots/',
help = 'path of folder where plots will be saved')
parser.add_argument('--disc', type = str, default = 'kl', choices = ['kl', 'hellinger', 'l1'],
help = 'discrepancy to estimate')
parser.add_argument('--target', type = str, default = 'cauchy', choices=['cauchy', 'mixture'],
help = 'target distribution to use')
parser.add_argument('--extension', type = str, default = 'pdf', choices = ['pdf', 'png', 'jpg'],
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
disc = args.disc



# IMPORT TARGET DENSITY ####
target = args.target
if target == 'cauchy':
    from targets.cauchy import *
    xlim = np.array([-13, 13]) # for plotting
if target == 'mixture':
    from targets.mixture import *
    xlim = np.array([-3, 15]) # for plotting

def p(x): return p_aux(x, 1)

# READ DATA FRAMES IN PATH ####
metadata = pd.DataFrame({'file_name': [], 'N': []})

for file_name in glob.glob(inpath + 'results' + '*.csv'):
    # read file and save info
    dat = pd.read_csv(file_name)
    metadata = metadata.append(pd.DataFrame({'file_name': [file_name], 'N': [dat.x.unique().shape[0]]}))
# end for


# PLOT ####
print('begin plotting!')
ss = metadata.N.unique() # save sample sizes
errors = pd.DataFrame({'N': [], 'disc': []})

# log densities
for N in ss:
    # files with that sample size
    files = metadata[metadata.N == N].file_name

    # initialize plot with target log density
    t = np.linspace(xlim[0], xlim[1], 2000)
    f = p(t[:, np.newaxis])
    plt.plot(t, f, 'k-', label = 'target', linewidth = 1, markersize = 1.5)
    i = 1

    for file in files:
        # read file and generate resulting approximation
        dat = pd.read_csv(file)
        q = nsvmi.q_gen(np.array(dat.w), np.array(dat.x), np.array(dat.rho))

        # plot log density estimation
        qN = q(t[:, np.newaxis])
        if i == 1: plt.plot(t, qN, 'c-', label = 'mixture', linewidth = 1, markersize = 0.75, alpha = 0.5)
        else: plt.plot(t, qN, 'c-', linewidth = 1, markersize = 0.75, alpha = 0.5)
        i += 1

        # calculate discrepancy for future plotting
        errors = errors.append(pd.DataFrame({'N': [N], 'disc': nsvmi.objective(p, q, np.array(dat.w), np.array(dat.x[:, np.newaxis]), np.array(dat.rho), B = 100000, type = disc)}))

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
    f = np.exp(p(t[:, np.newaxis]))
    plt.plot(t, f, 'k-', label = 'target', linewidth = 1, markersize = 1.5)
    i = 1

    for file in files:
        # read file and generate resulting approximation
        dat = pd.read_csv(file)
        q = nsvmi.q_gen(np.array(dat.w), np.array(dat.x), np.array(dat.rho))

        # plot log density estimation
        qN = np.exp(q(t[:, np.newaxis]))
        if i == 1: plt.plot(t, qN, 'c-', label = 'mixture', linewidth = 1, markersize = 0.75, alpha = 0.5)
        else: plt.plot(t, qN, 'c-', linewidth = 1, markersize = 0.75, alpha = 0.5)
        i += 1

        # calculate discrepancy for future plotting
        errors = errors.append(pd.DataFrame({'N': [N], 'disc': nsvmi.objective(p, q, np.array(dat.w), np.array(dat.x[:, np.newaxis]), np.array(dat.rho), B = 100000, type = disc)}))

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
plt.ylabel(disc)
plt.title('discrepancy boxplot')
plt.suptitle('')
plt.savefig(path + 'disc-plot.' + extension, dpi=900, bbox_inches='tight')

print('done plotting!')
