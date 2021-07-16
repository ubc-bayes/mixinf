# define an SMC sampler with a RWMH rejuvenation kernel

import numpy as np

def create_smc(sd, steps):
    # input:
    # sd    : standard deviation of the rwmh rejuvenation kernel
    # steps : number of steps taken by the chain at each rejuvenation step

    def logsumexp(x):
        m = np.amax(x,axis=-1)
        return m + np.log(np.sum(np.exp(x-m[...,np.newaxis]), axis=-1))

    #################
    #################
    # rwmh kernel ###
    #################
    #################
    def rwmh_step(x, logp):
        # input:
        # x     : (N,K) array with initial locations for sampling; N number of samples, K dimension
        # logp  : function with target log density
        #
        # output:
        # x     : (N,K) array with MCMC samples

        N = x.shape[0]
        K = x.shape[1]
        def r_sampler(y): return sd*np.random.randn(N,K) + y            # gaussian sampler
        for t in range(steps):
            tmp_x = r_sampler(x)                                        # generate proposal
            logratio = logp(tmp_x)-logp(x)                              # log hastings ratio
            swaps = np.log(np.random.rand(N)) < np.minimum(0, logratio) # indices with accepted proposals
            x[swaps,:] = tmp_x[swaps,:]                                 # update accepted proposals

        return x


    #################
    #################
    # smc sampler ###
    #################
    #################
    def smc(logp, logr, r_sample, B, beta_ls, Z0 = None):
        # input:
        # logp        : log density of target distribution
        # logr        : log density of reference distribution
        # r_sample    : function that generates a sample from reference distribution
        # B           : integer, number of particles
        # beta_ls     : (n_steps+1) array with the values of the beta partition (with beta_ls[0] = 0)
        # Z0          : None or float with normalizing constant of r
        #
        # output:
        # x           : (N,K) array with sample points (K is dimension)
        # Z           : if Z is None, Z1/Z0; else, Z1, an estimate of the normalizing constant of p
        # ESS         : effective sample size


        x = r_sample(B)                # init sample points
        n_steps = beta_ls.shape[0]-1   # take 1 off to account for first beta
        logm = lambda x : logr(x)      # first target is the reference
        logZ = np.zeros(n_steps)       # for tracking normalizing constant log ratios
        ESS = None

        for k in range(n_steps):
            new_beta = beta_ls[k+1]
            new_logm = lambda x : (1-new_beta)*logr(x) + new_beta*logp(x)

            ##### reweight  #####
            logw = new_logm(x) - logm(x)                      # importance weight
            logZ[k] = logsumexp(logw - np.log(logw.shape[0])) # log normalizing constant ratio
            w = np.exp(logw)
            ESS = (np.sum(w))**2 / np.sum(w**2)               # keep track of this
            w = w/np.sum(w)                                   # normalize for np.random.multinomial

            ##### resample   #####
            reps = np.random.multinomial(n = x.shape[0], pvals = w)
            tmp_x = np.repeat(x, reps, axis=0).reshape(x.shape)

            ##### rejuvenate #####
            x = rwmh_step(x = tmp_x, logp = new_logm)

            ##### update     #####
            beta = beta_ls[k+1]
            logm = lambda x : (1-beta)*logr(x) + beta*logp(x)
        # end for

        Z = np.exp(np.sum(logZ))
        if Z0 is not None: Z = Z*Z0

        return x,Z,ESS

    return smc
