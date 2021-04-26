import numpy as np


# Gaussian proposals
sd = 0.5
sd = 0.25
def logr(x, y): return -0.5 * x.shape[1] * np.sum((x - y)**2, axis = -1) / sd**2 - 0.5*x.shape[1]*np.log(2*np.pi) - x.shape[1]*np.log(sd) # gaussian log density
def r_sampler(T, y): return sd * np.random.randn(T, y.shape[1]) + y # gaussian sampler

# mcmc kernel sampler
def kernel_sampler(y, T, S, logp):
    # shape(N,K) array of locations y
    # shape(N,) array with no. steps per location T
    # sample size S
    # target logdensity logp
    #
    # out: array of shape(S, N, K)

    N = y.shape[0]
    K = y.shape[1]
    y = y.astype(float)
    out = np.tile(y, (S,1))   # generates a shape(S*N,K) array
    M = out.shape[0] # M = S*N

    steps = np.repeat(T, S)
    max_T = T.max().astype(int) # max number of steps to take
    running = np.arange(M) # chains still running

    for t in range(max_T):

        no_running = running.shape[0] # how many chains are still being run?
        tmp_x = r_sampler(no_running, out[running, :]) # generate proposal
        #print('proposal: ' + str(tmp_x))

        logratio = logp(tmp_x) + logr(tmp_x, out[running, :]) - logp(out[running, :]) - logr(out[running, :], tmp_x) # log hastings ratio
        logu = np.log(np.random.rand(no_running))
        #print('log ratio: ' + str(logratio))


        tmp_out = out[running, :]
        swaps = logu < np.minimum(0, logratio) # indices of accepted proposals
        tmp_out[swaps, :] = tmp_x[swaps, :] # substitute values
        out[running, :] = tmp_out

        # update active chains
        running = np.arange(M)[t+1 <= steps] # chains in which less than the max no. of steps have been taken

    # end for

    return out.reshape(S, N, K)
