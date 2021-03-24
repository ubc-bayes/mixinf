import numpy as np


# Gaussian proposals
sd = 0.5
def logr(x, y): return -0.5 * (x - y)**2 / sd**2 - 0.5*np.log(2*np.pi) - np.log(sd) # gaussian log density
def r_sampler(T, y): return sd * np.random.randn(T) + y # gaussian sampler

# mcmc kernel sampler
def kernel_sampler(y, T, N, logp):
    # array of locations y
    # array with no. steps per location T
    # sample size N
    # target logdensity logp
    #
    # out: array of shape(N, y.shape[0])

    y = y.astype(float)
    out = np.repeat(y, N)   # generates a shape(M,) array
    M = out.shape[0]

    steps = np.repeat(T, N)
    max_T = T.max().astype(int) # max number of steps to take
    running = np.arange(M) # chains still running

    for t in range(max_T):


        no_running = running.shape[0] # how many chains are still being run?
        tmp_x = r_sampler(no_running, out[running]) # generate proposal

        logratio = logp(tmp_x) + logr(tmp_x, out[running]) - logp(out[running]) - logr(out[running], tmp_x) # log hastings ratio
        logu = np.log(np.random.rand(no_running))


        tmp_out = out[running]
        swaps = logu < np.minimum(0, logratio) # indices of accepted proposals
        tmp_out[swaps] = tmp_x[swaps] # substitute values
        out[running] = tmp_out

        # update active chains
        running = np.arange(M)[t+1 <= steps] # chains in which less than the max no. of steps have been taken

    # end for

    return out.reshape(y.shape[0], N).T
