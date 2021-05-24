import numpy as np
from mc_theta import adaptive_truncation_sampler, rej_beta
import pandas as pd
import os

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

# print the processed data
#print('Data:')
#print(df)

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
np.random.seed(3)
T = 2000
K = 2010
out = np.zeros((7,K+3))

# sample
Alphs, Gams, Lambs, Ths = adaptive_truncation_sampler(T = T, Edges = Obs, N = N, K = K,
                                alph=0.1, lamb=20., gam=2.,
                                alpha_step = 0.04, lambda_step = 0.1,
                                Th0_step = 0.03, Th1_step = 0.1, ThK_step = 0.1)

# save in array
out[:,0] = Alphs
out[:,1] = Gams
out[:,2] = Lambs
out[:,3:] = Ths

# save array
np.save('../mcmc_sample.npy', out)
