import numpy as np


# Gaussian proposals
sd = 1
#sd = 0.25
def logr(x, y): return -0.5*np.sum((x - y)**2, axis = -1)/sd**2 - 0.5*x.shape[1]*np.log(2*np.pi) - 0.5*np.log(sd) # gaussian log density
def r_sampler(T, y): return sd * np.random.randn(T, y.shape[1]) + y # gaussian sampler

# mcmc kernel sampler
def gaussian_sampler(y, T, S, logp):
    # this function simulates NxS RWMH chains for T steps
    #
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

def kernel_sampler(y, T, S, logp, t_increment, chains = None, update = False):
    # this function determines the incremental runs needed and calls the above gaussian sampler
    #
    # shape(N,K) array of locations y
    # shape(N,) array with no. steps per location T
    # sample size S
    # target logdensity logp
    # t_increment is the incremental steps taken at each iteration per kernel
    # chains is a list of N shape(T_n,K) arrays with the current chains
    #   it is used for cacheing. If None, then the whole chain is generated from scratch
    # update is boolean. If True and chains is not None, a new chains list will be created with the generated samples appended
    #
    # out: array of shape(S, N, K)
    
    # if chain has to be run from scratch, call gaussian sampler with full T
    if chains is None:
        return gaussian_sampler(y, T, S, logp)
    else:
        # determine the current T in the chains
        N = y.shape[0]
        K = y.shape[1]
        Tnew = np.zeros(N)
        ynew = np.zeros((N,K))
        #print('chains: ' + str(chains))
        for n in range(N):
            # determine incremental T's
            Tcurr = t_increment*(chains[n].shape[0] - 1) # -1 to account for initial sample
            Tnew[n] = T[n] - Tcurr # chains[n] is shape(T_n,K)
            if Tnew[n] < 0:
                print('error! negative incremental steps')
                print('n = ' + str(n))
                print('kernel sampler y: ' + str(y))
                print('y = ' + str(y[n,:]))
                print('kernel sampler T: ' + str(T))
                print('kernel sampler chains: ' + str(chains))
                print('current length: ' + str(Tcurr))
                print('needed length: ' + str(T[n]))
                print('incremental steps: ' + str(Tnew[n]))

            # update starting points to the previous iter (to account for +1)
            ynew[n,:] = chains[n][chains[n].shape[0]-1,:]
        # end for

        # now call gaussian sampler with the incremental samples only
        #print('sampling ' + str(Tnew) + ' instead of ' + str(T))
        samples = gaussian_sampler(ynew, Tnew+1, S, logp) #(S,N,K)

        if update:
            # append first sample to each chain
            for n in range(N):
                if Tnew[n] > 0: chains[n] = np.vstack((chains[n], samples[0,n,:].reshape(1,K)))
            # end for

            return samples, chains
        else:
            return samples
