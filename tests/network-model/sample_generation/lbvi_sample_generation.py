import numpy as np
import pandas as pd
#from kernel import kernel_sampler, rej_beta
#from kernel import *
import sys
import os
#from sampler import adaptive_sampler, sampler_wrapper
from samplerv2 import adaptive_truncation_sampler

sys.path.insert(1, os.path.join(sys.path[0], '../../kernels/'))

#################
#################
## data import ##
#################
#################


# load the network
df = pd.read_csv('../fb2.txt', delim_whitespace=True, )

# convert the datetime to integers
df['rnd'] = pd.to_datetime(df['date']).astype(int)/(1800*10**9) # this converts to 1/2 hour intervals
df['rnd'] = (df['rnd'] - df['rnd'][0]).astype(int) #this shifts the intervals by the first time

# remove any self loop entries
df = df[df.v1 != df.v2]

# shift the indices down by 1 (start at 0)
df.v1 -= 1
df.v2 -= 1

# convert the relevant columns to numpy arrays
v1 = np.array(df.v1)
v2 = np.array(df.v2)
rnds = np.array(df.rnd)

# threshold the rounds

rnd_begin = 3950
rnd_end = 10377

v1 = v1[(rnds >= rnd_begin) & (rnds < rnd_end)]
v2 = v2[(rnds >= rnd_begin) & (rnds < rnd_end)]
rnds = rnds[(rnds >= rnd_begin) & (rnds < rnd_end)]


# collect the number of edges and vertices binned into days
# censor to the range when the first edge appears, and before a shift in rate happens
nE = [0] # because we use nE[-1] below; we will remove this entry right after
nV = []
nN = []
V = set()
for n in range(rnd_begin, rnd_end):
    idcs = np.where(rnds == n)[0]
    nE.append(nE[-1] + idcs.shape[0])
    for i in idcs:
        V.add(v1[i])
        V.add(v2[i])
    nV.append(len(V))
    nN.append(n+1)
nE = np.array(nE)[1:] #remove the extra first entry we put before
nV = np.array(nV)
nN = np.array(nN)

# collect the edges into a sparse representation
N = rnd_end - rnd_begin
K = max(v1.max(), v2.max())+1
X = np.zeros((K,K))
for i in range(v1.shape[0]):
    X[v1[i], v2[i]] += 1
degrees = (X+X.T).sum(axis=0)
hist, edges = np.histogram(degrees, density=True, bins=50)
edges = edges[1:]

idcs = np.where(X>0)
Obs = np.zeros((3, idcs[0].shape[0]), dtype=np.int64)
Obs[0,:] = idcs[0]
Obs[1,:] = idcs[1]
Obs[2,:] = X[idcs[0], idcs[1]]



#################
#################
## sample generation
#################
#################

# init
np.random.seed(123)
#T = 1000*np.ones(7)
T = 1000
K = 2010
out = np.zeros((7,K+3))
#alpha_init = np.array([0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9]).reshape(7,1)
alpha_init = np.array([0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9])
#gam_init = 2.*np.ones((7,1))
#lamb_init = 20.*np.ones((7,1))
#Th_init = np.full((7,K), None)
#y = np.hstack((alpha_init, gam_init, lamb_init, Th_init))


# sample
for i in range(alpha_init.shape[0]):
    Alphs, Gams, Lambs, Ths = adaptive_truncation_sampler(T, Obs, N, K,
                                    alph=alpha_init[i], lamb=20., gam=2.,
                                    alpha_step = 0.04, lambda_step = 0.1,
                                    Th0_step = 0.03, Th1_step = 0.1, ThK_step = 0.1)

    # save in array
    out[i,0] = Alphs[-1]
    out[i,1] = Gams[-1]
    out[i,2] = Lambs[-1]
    out[i,3:] = Ths[-1,:]

#out = sampler_wrapper(y, T, S = 1, logp = None)

# save array
np.save('../initial_sample.npy', out)
