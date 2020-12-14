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
sys.path.insert(1, os.path.join(sys.path[0], '../../mixinf/'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import nsvmi # functions to do normal seq-opt variational mixture inference

# ARG PARSE SETTINGS ####
parser = argparse.ArgumentParser(description="plot results from normal-kernel bivariate variational mixture inference")

parser.add_argument('--inpath', type = str, default = 'results/',
help = 'path of folder where csv files are stored')
parser.add_argument('--outpath', type = str, default = 'results/plots/',
help = 'path of folder where plots will be saved')
parser.add_argument('--disc', type = str, default = 'kl', choices = ['kl', 'hellinger', 'l1'],
help = 'discrepancy to estimate')
parser.add_argument('--target', type = str, default = 'cauchy', choices=['cauchy', 'mixture', 'banana'],
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
    xlim = np.array([-30, 30]) # for plotting
    ylim = np.array([-30, 30]) # for plotting
if target == 'mixture':
    from targets.mixture import *
    xlim = np.array([-3, 15]) # for plotting
    ylim = np.array([-3, 15]) # for plotting
if target == 'banana':
    from targets.banana import *
    xlim = np.array([-60, 60]) # for plotting
    ylim = np.array([-60, 80]) # for plotting

def p(x): return p_aux(x, 2)

# READ DATA FRAMES IN PATH ####
metadata = pd.DataFrame({'file_name': [], 'N': []})

for file_name in glob.glob(inpath + 'results' + '*.csv'):
    # read file and save info
    dat = pd.read_csv(file_name)
    metadata = metadata.append(pd.DataFrame({'file_name': [file_name], 'N': [len(dat.index)]}))
# end for


# PLOT ####
print('begin plotting!')
errors = pd.DataFrame({'N': [], 'disc': []})


for file in metadata.file_name:

    # read data and generate mixture
    dat = pd.read_csv(file)                # read data
    dat = dat[dat.w > 0]                   # drop weights that are zero
    w = np.array(dat.w)                    # get weights
    w = w / np.sum(w)                      # normalize
    rho =  np.array(dat.rho)               # get sd
    dat = dat.drop(['w', 'rho'], axis = 1) # remove weights and sd
    x = np.array(dat)                      # remaining columns are the data
    N = w.shape[0]                         # mixture size
    q = nsvmi.q_gen(w, x, rho)             # get mixture

    # initialize plot values
    xx = np.linspace(xlim[0], xlim[1], 2000)
    yy = np.linspace(ylim[0], ylim[1], 2000)
    tt = np.array(np.meshgrid(xx, yy)).T.reshape(2000**2, 2)
    f = np.exp(q(tt)).reshape(2000, 2000).T

    # plot
    plt.contour(xx, yy, f)
    plt.scatter(x[:, 0], x[:, 1], s = 500*w, marker='.', c='k')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # save plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('mixture contour plot for N = ' + str(N))
    title = 'contour_N' + str(N) + '.'
    plt.savefig(path + title + extension, dpi=900)
    plt.clf()


    # calculate discrepancy for future plotting
    errors = errors.append(pd.DataFrame({'N': [N], 'disc': nsvmi.objective(p, q, w, x, rho, B = 100000, type = disc)}))
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
