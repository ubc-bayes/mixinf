# Mixture kernels

Each script in this directory contains a different mixture kernel. Specifically, each scripts should define a function `kernel_sampler(y, T, S, logp)`. The function generates `N x S` `K`-dimensional chains in parallel and runs chain `i` for `T[i]` steps.

Here,
* `y` is a `shape(N,K)` array with kernel locations (`N` is sample size, `K` is dimension of problem)
* `T` is a `shape(N,)` array with the number of steps to run each of the `N` chains
* `S` is an integer that indicates the sample size of each sample (in total, `N x S` chains are run in parallel)
* `logp` is the target log density

### Directory roadmap
* `gaussian.py` contains a random-walk Metropolis-Hastings sampler with symmetric Gaussian proposals
